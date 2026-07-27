import os, sys, time, h5py
import numpy as np
import open3d as o3d

def class_id_to_one_hot(class_id, num_classes=41):
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[class_id] = 1.0
    return one_hot


class PointCloudProcessor:
    def __init__(self, points):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)

    def get_bounding_box(self):
        """
        Computes the oriented bounding box based on the PCA of the convex hull.
        """
        try:
            # Try computing the Oriented Bounding Box (OBB)
            obb = self.pcd.get_oriented_bounding_box()
            position = obb.center
            extent = obb.extent
            rotation = obb.R  # 3x3 rotation matrix
        except RuntimeError:
            print("Warning: OBB computation failed, using AABB instead.")
            
            # Compute Axis-Aligned Bounding Box (AABB)
            aabb = self.pcd.get_axis_aligned_bounding_box()
            max_pos = aabb.max_bound
            min_pos = aabb.min_bound
            position = (max_pos + min_pos) / 2
            extent = np.abs(max_pos - min_pos)
            rotation = np.eye(3)  # Identity matrix (no rotation)

        return position, extent, rotation

    def process(self, voxel_size=0.01):
        # sparsify pcl
        self.pcd = self.pcd.voxel_down_sample(voxel_size)
        # remove outlier
        nb_neighbors=10
        std_thres=2.0
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors, std_thres)
        # calculate boundiing box
        return self.get_bounding_box()

class Segment:
    def __init__(self, points, is_thing, instance_label, class_label, \
            inst_confidence, overlap_ratio, pose, index, center = None, segment_label = 0, sem_feat=None):

        self.points = points.astype(np.float32).reshape(-1,3)
        self.is_thing = is_thing
        self.instance_label = np.float32(instance_label)
        self.class_label= class_label
        # new for open-set semantic
        self.sem_feat = sem_feat
        self.inst_confidence = np.float32(inst_confidence)
        self.overlap_ratio = np.float32(overlap_ratio)
        self.pose = pose.astype(np.float32)
        self.index = index
        self.geometry_confidence = None
        self.segment_label = segment_label

        if (center is None) or (np.array(center).shape != (1,3)):  
            self.center = np.mean(self.points, axis=0).astype(np.float32).reshape(1,3)
        else:
            self.center = center.astype(np.float32).reshape(1,3)
        self.box_points = np.zeros((1,3))

    def calculateConfidenceDefault(self, weight=0.5):
        self.geometry_confidence = np.ones((1,self.points.shape[0])).reshape(1,-1).astype(np.float32)

    def calculateBBox(self, voxel_grid = 0.02, sampling_dist = 0.025):
        # seg_pcl = pcl.PointCloud(self.points)
        # # sparsify pcl
        # pcl_sparse_filter = seg_pcl.make_voxel_grid_filter()
        # pcl_sparse_filter.set_leaf_size(voxel_grid,voxel_grid,voxel_grid)
        # seg_pcl_voxel = pcl_sparse_filter.filter()
        # # remove outlier
        # pcl_filter = seg_pcl_voxel.make_statistical_outlier_filter()
        # pcl_filter.set_mean_k (10)
        # pcl_filter.set_std_dev_mul_thresh (2.0)
        # pcl_filtered =  pcl_filter.filter()
        # # calculate boundiing box
        # Bbox_extractor = pcl.MomentOfInertiaEstimation()
        # Bbox_extractor.set_InputCloud(pcl_filtered)
        # Bbox_extractor.compute()
        # min_point_OBB, max_point_OBB, position_OBB, rot_matrix_OBB = Bbox_extractor.get_OBB()
        
        # ###################### New with Open3D ######################
        PCLprocessor = PointCloudProcessor(self.points)
        bbox_res = PCLprocessor.process(voxel_size=voxel_grid)
        bbox_pos, bbox_extent, bbox_rot_matrix = bbox_res

        # print("Size of Bbox:", OBB.extent, ", with sample dist:", sampling_dist)
        # [1, 3]
        extent_half = np.array(bbox_extent).reshape(1, -1) / 2.0

        sampling_num = np.maximum(
            np.floor(2 * extent_half / sampling_dist - 1), 1
        )[0].astype(int)

        # Generate sample ranges
        x_sample_range = np.linspace(-1, 1, sampling_num[0], endpoint=False)
        y_sample_range = np.linspace(-1, 1, sampling_num[1], endpoint=False)
        z_sample_range = np.linspace(-1, 1, sampling_num[2], endpoint=False)

        # Base unit vertices
        Bbox_vertices_unit = np.array([
            [0, 0, 0],  # box center
            [1, 1, 1], [-1, 1, 1], [-1, 1, -1], [1, 1, -1],
            [1, -1, 1], [-1, -1, 1], [-1, -1, -1], [1, -1, -1]
        ])

        # Generate three primary edges
        edge_x0 = np.stack([x_sample_range, 
            np.ones_like(x_sample_range), np.ones_like(x_sample_range)
        ]).T # v1-v2
        edge_y0 = np.stack([np.ones_like(y_sample_range), 
            y_sample_range, np.ones_like(y_sample_range)
        ]).T # v1-v5
        edge_z0 = np.stack([np.ones_like(z_sample_range), 
            np.ones_like(z_sample_range), z_sample_range
        ]).T # v4-v1

        edge_x1 = edge_x0 * np.array([[1, 1, -1]]) # v3-v4
        edge_x2 = edge_x0 * np.array([[1, -1, -1]]) # v7-v8
        edge_x3 = edge_x0 * np.array([[1, -1, 1]]) # v5-v6
        edge_y1 = edge_y0 * np.array([[1, 1, -1]]) # v4-v8
        edge_y2 = edge_y0 * np.array([[-1, 1, -1]]) # v3-v7
        edge_y3 = edge_y0 * np.array([[-1, 1, 1]]) # v2-v6
        edge_z1 = edge_z0 * np.array([[1, -1, 1]]) # v8-v5
        edge_z2 = edge_z0 * np.array([[-1, -1, 1]]) # v6-v7
        edge_z3 = edge_z0 * np.array([[-1, 1, 1]]) # v2-v3
        # (N, 3)
        Bbox_vertices_unit = np.vstack([Bbox_vertices_unit, 
            edge_x0, edge_x1, edge_x2, edge_x3, 
            edge_y0, edge_y1, edge_y2, edge_y3, 
            edge_z0, edge_z1, edge_z2, edge_z3
        ])
        Bbox_vertices_AA = Bbox_vertices_unit * np.repeat(
            np.abs(extent_half), Bbox_vertices_unit.shape[0], axis=0
        ) 
        Bbox_vertices_Orient = (Bbox_vertices_AA @ bbox_rot_matrix.T) + bbox_pos.reshape(1,3)
        self.box_points = (Bbox_vertices_Orient).reshape(-1,3).astype(np.float32)


def checkSegmentFramesEqual(segs_framesA, segs_framesB):
    if(len(segs_framesA)!=len(segs_framesB)):
        print(" Not Equal, length of segment lists")
        return False
    for f_i, seg_A in enumerate(segs_framesA):
        seg_B = segs_framesB[f_i]
        # print("     check seg num %d "%(f_i))
        if(not np.isclose(seg_A.pose,seg_B.pose).all()):
            print("    Not Equal pose in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.center,seg_B.center).all()):
            print("    Not Equal center in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.points,seg_B.points, atol=1e-4).all()):
            print("    Not Equal points in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.inst_confidence,seg_B.inst_confidence).all()):
            print("    Not Equal label_confidence in %d frame "%(f_i))
            return False
        if(not np.isclose(seg_A.overlap_ratio,seg_B.overlap_ratio).all()):
            print("    Not Equal label_confidence in %d frame "%(f_i))
            return False
        if((seg_A.instance_label!=seg_B.instance_label)):
            print("    Not Equal instance_label in %d frame "%(f_i))
            return False
        if((seg_A.class_label!=seg_B.class_label)):
            print("    Not Equal class_label in %d frame "%(f_i))
            return False
        if((seg_A.is_thing!=seg_B.is_thing)):
            print("    Not Equal class_label in %d frame "%(f_i))
            return False
    return True

def isPoseValid(pose):
    is_nan = np.isnan(pose).any() or np.isinf(pose).any()
    if is_nan:
        return False

    R_matrix = pose[:3, :3]
    I = np.identity(3)
    is_rotation_valid = ( np.isclose( np.matmul(R_matrix, R_matrix.T), I , atol=1e-3) ).all and np.isclose(np.linalg.det(R_matrix) , 1, atol=1e-3)
    if not is_rotation_valid:
        return False

    return True


def dictToHd5(file, dict):
	f = h5py.File(file,'w')
	for key in dict:
		f[key] = dict[key]
	f.close() 
	return None

def hd5ToDict(file):
	f = h5py.File(file,'r')
	dict = {}
	for key in f:
		dict[key] = np.array(f[key])
	f.close() 
	return dict


def getHostId(gpu_id):
    HOST = str(gpu_id) + ".0.0."+str(gpu_id)
    return HOST

def getGpuDevice(gpu_id, gpu_num):
    assert(gpu_id >= 0)
    assert(gpu_num > gpu_id)
    if(gpu_num == 1):
        return "cuda"
    return "cuda:"+str(gpu_id)



def get_new_pallete(num_colors):
    """Generate a color pallete given the number of colors needed. First color is always black."""
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return np.array(pallete).astype(np.uint8)
