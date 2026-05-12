import os, copy, glob, json
from os.path import join as pjoin
import pandas as pd
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import cv2
from .common_utils import isPoseValid


class BaseLoader:
    def __init__(self, dir, preload_img = False, preload_depth = False):
        # whether to preload data into memory
        self.preload_img = preload_img
        self.preload_depth = preload_depth

        self.dir = dir
        self.rgb_path_map = ...
        self.depth_path_map = ...
        self.poses = ...
        self.rgb_intrinsic = ...
        # self.rgb_h = ...
        # self.rgb_w = ...
        self.depth_intrinsic = ...
        self.depth_h = ...
        self.depth_w = ...
        self.depth_scale = 1.0
        self.indexes = ...
        self.index_min = 0

        if preload_img:
            self.images = ...
        if preload_depth:
            self.depths = ...

    def getRGBCameraMatrix(self):
        # return rgb camera matrix 
        return self.rgb_intrinsic.astype(np.float32)
    
    def getDepthCameraMatrix(self):
        # return depth camera matrix 
        return self.depth_intrinsic.astype(np.float32)
    
    def getPoseFromIndex(self, index):
        pose = self.poses[index]
        return pose
    
    mapRGBtoDepth = None

    def getDataFromIndex(self, index):
        """Get BGR image, depth image and pose from index.
        Args:
            index (int): The index of the data to be retrieved.
        Returns:
            BGR Image aligned with depth camera.
        """
        if index == -1:
            index = self.index_min
        if(index not in self.indexes):
            return None,None,None

        rgb_img_aligned = None
        depth_img = None
        pose = None

        if self.preload_img:
            rgb_img_aligned = self.images[index]
        else:
            image_f = self.rgb_path_map[index]
            rgb_img = cv2.imread(image_f, cv2.IMREAD_UNCHANGED)
            if self.mapRGBtoDepth is not None:
                rgb_img_aligned = self.mapRGBtoDepth(rgb_img)
            else:
                rgb_img_aligned = rgb_img

        if self.preload_depth:
            depth_img = self.depths[index]
        else:
            depth_f = self.depth_path_map[index]
            depth_img = cv2.imread(depth_f, cv2.IMREAD_UNCHANGED)
        depth_img_scaled = depth_img.astype(np.float32) / self.depth_scale
        
        pose = self.poses[index].astype(np.float32)
        # check validity of pose
        is_pose_valid = isPoseValid(pose)
        if not is_pose_valid:
            return None, None, None

        return rgb_img_aligned, depth_img_scaled, pose


class ScannetLoader(BaseLoader):
    def __init__(self, dir, preload_img = False, preload_depth = False):
        super().__init__(dir, preload_img, preload_depth)

        # parse data location
        depth_folder = pjoin(self.dir, "depth")
        self.depth_files = sorted(os.listdir(depth_folder))

        self.depth_indexes = [int(depth_f.split('.')[0]) for depth_f in self.depth_files]
        self.depth_path_map = {
            index: os.path.join(depth_folder, str(index)+".png"
                ) for index in self.depth_indexes
        }
        
        self.rgb_folder = os.path.join(self.dir, "color")
        self.rgb_files = sorted(os.listdir(self.rgb_folder))

        self.rgb_indexes = [int(color_f.split('.')[0]) for color_f in self.rgb_files]
        self.rgb_path_map = {
            index: os.path.join(self.rgb_folder, str(index)+".jpg"
                ) for index in self.rgb_indexes
        }

        # load poses first
        self.pose_folder = os.path.join(self.dir, "pose")
        self.pose_files = sorted(os.listdir(self.pose_folder))

        self.readPoses()
        self.traj_indexes = list(self.poses.keys())

        # get frame indexs
        self.indexes = set.intersection( set(self.depth_indexes), set(self.rgb_indexes),  set(self.traj_indexes))
        self.indexes = list(self.indexes)
        self.indexes.sort()
        self.index_min = min(self.indexes)
        self.index_max = max(self.indexes)
        print(f"Indexes: from {self.index_min} to {self.index_max}, len {len(self.indexes)}")

        # get camera matrixs
        self.rgb_intrinsic_f = os.path.join(
            self.dir, "intrinsic", "intrinsic_color.txt")
        self.depth_intrinsic_f = os.path.join(
            self.dir, "intrinsic", "intrinsic_depth.txt")
        self.rgb_intrinsic = np.loadtxt(self.rgb_intrinsic_f)[:3, :3]
        self.depth_intrinsic = np.loadtxt(self.depth_intrinsic_f)[:3, :3]
        self.depth_scale = 1000.0
        self.homograph_color_to_depth = self.depth_intrinsic @ np.linalg.inv(self.rgb_intrinsic)

        # get depth image shape 
        depth_f = self.depth_path_map[self.indexes[0]]
        depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        self.depth_h = depth_img.shape[0]
        self.depth_w = depth_img.shape[1]

        # add the gt 2D semntics
        # label convertion map
        label_df = pd.read_csv(f"{dir}/../scannetv2-labels.combined.tsv", sep="\t")
        self.label_mapping = dict(zip(label_df["id"], label_df["nyu40id"]))
        self.gt_sem_folder = '/home/zilong/Downloads/scans/scene0001_00/2d-label-filt'
        self.gt_inst_folder = '/home/zilong/Downloads/scans/scene0001_00/2d-instance-filt'

        # preload data in RAM
        if self.preload_img:
            self.images = {}
            for idx in self.indexes:
                image_f = self.rgb_path_map[idx]
                rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
                rgb_img_aligned = cv2.warpPerspective(rgb_img, self.homograph_color_to_depth,
                    (self.depth_w, self.depth_h) )
                self.images[idx] = rgb_img_aligned

        if self.preload_depth:
            self.depths = {}
            for idx in self.indexes:
                depth_f = self.depth_path_map[idx]
                depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
                self.depths[idx] = depth_img

    def readPoses(self):
        self.poses = {}
        for i, pose_f in enumerate(self.pose_files):
            T_WC = np.loadtxt(os.path.join(self.pose_folder, pose_f))
            if isPoseValid(T_WC):
                # normalization
                r = Rotation.from_matrix(T_WC[:3,:3])
                T_WC[:3,:3] = r.as_matrix()
            # add to the dict
            pose_index = int(pose_f.split('.')[0])
            self.poses[pose_index] = T_WC

    def mapRGBtoDepth(self, rgb_img):
        return cv2.warpPerspective(
            rgb_img, self.homograph_color_to_depth,
            (self.depth_w, self.depth_h)
        )

    def getGTSemanticFromIndex(self, index):
        if index == -1:
            index = self.index_min
        if(index not in self.indexes):
            return None
        
        gt_sem_segs = None
        sem_f = os.path.join(self.gt_sem_folder, str(index)+".png")
        gt_sem_segs = cv2.imread(sem_f, cv2.IMREAD_UNCHANGED)
        # map all label ids to categories id
        gt_labels = np.unique(gt_sem_segs)
        for label in gt_labels:
            if label == 0:
                continue
            if label in self.label_mapping:
                gt_sem_segs[gt_sem_segs==label] = self.label_mapping[label]
        return gt_sem_segs

    def getGTInstanceFromIndex(self, index):
        if index == -1:
            index = self.index_min
        if(index not in self.indexes):
            return None
        
        gt_inst_segs = None
        inst_f = os.path.join(self.gt_inst_folder, str(index)+".png")
        gt_inst_segs = cv2.imread(inst_f, cv2.IMREAD_UNCHANGED)
        return gt_inst_segs
        
    
    
    def calculatePoseConfidence(self):
        # inlier_ratio_arr = np.loadtxt(self.inlier_ratio_f).reshape(-1,2)
        # self.inlier_ratio_map = {}
        # for arr_idx in range(inlier_ratio_arr.shape[0]):
        #     self.inlier_ratio_map[int(inlier_ratio_arr[arr_idx,0])] = inlier_ratio_arr[arr_idx,1]
        
        # # calculate pose confidence, try inlier ratio first 
        # self.pose_confidence_map = {}
        # for index in self.indexes:
        #     if index == self.index_min:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     if index not in self.inlier_ratio_map:
        #         self.pose_confidence_map[index] = 0.5
        #         continue
        #     else:
        #         self.pose_confidence_map[index] = self.inlier_ratio_map[index]
            
        # uncertainty_arr = np.loadtxt(self.uncertainty_f).reshape(-1,2)
        # mean = np.mean(uncertainty_arr[:,1])
        # std = np.std(uncertainty_arr[:,1])
        
        # self.uncertainty_map = {}
        # for arr_idx in range(uncertainty_arr.shape[0]):
        #     self.uncertainty_map[int(uncertainty_arr[arr_idx,0])] = uncertainty_arr[arr_idx,1]
        
        # # calculate pose confidence, try inlier ratio first 
        # self.pose_confidence_map = {}
        # for index in self.indexes:
        #     if index == self.index_min:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     if index not in self.uncertainty_map:
        #         self.pose_confidence_map[index] = 1.0
        #         continue
        #     else:
        #         confidence = 0.8 + (self.uncertainty_map[index] - mean)/(2*std)
        #         confidence = np.clip(confidence, 0.0, 1.0)
        #         self.pose_confidence_map[index] = confidence
        
        if not os.path.isfile(self.inlier_num_f):
            self.pose_confidence_map = {}
            for index in self.indexes:
                self.pose_confidence_map[index] = 1.0
            return
        inlier_num_arr = np.loadtxt(self.inlier_num_f) 
        self.inlier_num = {}
        for arr_idx in range(inlier_num_arr.shape[0]):
            self.inlier_num[int(inlier_num_arr[arr_idx,0])] = inlier_num_arr[arr_idx,1]
        # calculate pose confidence, try scaled inlier num 
        self.pose_confidence_map = {}
        for index in self.indexes:
            if index == self.index_min:
                self.pose_confidence_map[index] = 2.0
                continue
            if index not in self.inlier_num:
                self.pose_confidence_map[index] = 1.0
                continue
            else:
                confidence = 1.0 + self.inlier_num[index] /1000.0
                confidence = min(confidence, 3.0)
                self.pose_confidence_map[index] = confidence
                
        breakpoint =  None
    

    def readPoseConfidence(self, index):
        if index not in self.pose_confidence_map:
            return None
        else:   
            return self.pose_confidence_map[index]
    

class ReplicaLoader(BaseLoader):
    def __init__(self, dir, preload_img = False, preload_depth = False):
        super().__init__(dir, preload_img, preload_depth)

        # parse data location
        data_folder = pjoin(self.dir, "results")

        self.depth_files = sorted(glob.glob(pjoin(data_folder, "depth*.png")))
        self.depth_indexes = [int(os.path.basename(depth_f).split('.')[0][-6:]) for depth_f in self.depth_files]
        self.depth_path_map = {
            index: pjoin(data_folder, f"depth{index:06d}.png") for index in self.depth_indexes
        }
        
        self.rgb_files = sorted(glob.glob(pjoin(data_folder, "frame*.jpg")))
        self.rgb_indexes = [int(os.path.basename(color_f).split('.')[0][-6:]) for color_f in self.rgb_files]
        self.rgb_path_map = {
            index: pjoin(data_folder, f"frame{index:06d}.jpg") for index in self.rgb_indexes
        }

        # load poses first
        self.traj_f = pjoin(self.dir, 'traj.txt')
        self.readPosesFromTraj()
        self.traj_indexes = list(self.poses.keys())

        # get frame indexs
        self.indexes = set.intersection(
            set(self.depth_indexes), set(self.rgb_indexes),  set(self.traj_indexes))
        self.indexes = sorted(list(self.indexes))
        self.index_min = min(self.indexes)
        self.index_max = max(self.indexes)
        print(f"Indexes: from {self.index_min} to {self.index_max}, len {len(self.indexes)}")

        # get camera matrixs
        self.depth_intrinsic_f = pjoin(os.path.dirname(dir), "cam_params.json")
        # load json
        with open(self.depth_intrinsic_f, 'r') as f:
            cam_params = json.load(f)['camera']
        self.depth_intrinsic = np.eye(3)
        self.depth_intrinsic[0, 0] = cam_params['fx']
        self.depth_intrinsic[1, 1] = cam_params['fy']
        self.depth_intrinsic[0, 2] = cam_params['cx']
        self.depth_intrinsic[1, 2] = cam_params['cy']
        self.depth_scale = cam_params['scale']
        self.depth_h = cam_params['h']
        self.depth_w = cam_params['w']

        # preload data in RAM
        if self.preload_img:
            self.images = {}
            for idx in self.indexes:
                image_f = self.rgb_path_map[idx]
                rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
                self.images[idx] = rgb_img

        if self.preload_depth:
            self.depths = {}
            for idx in self.indexes:
                depth_f = self.depth_path_map[idx]
                depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
                self.depths[idx] = depth_img

    def readPosesFromTraj(self):
        trajs = np.loadtxt(self.traj_f)
        trajs = trajs.reshape(-1, 4, 4)
        self.poses = {
            i: trajs[i] for i in range(trajs.shape[0])
        }




class SceneNNLoader:
    def __init__(self, dir, traj_filename):
        self.dir = dir
        self.depth_folder = os.path.join(self.dir, "depth")
        self.rgb_folder = os.path.join(self.dir, "image")
        self.depth_files = os.listdir(self.depth_folder)
        self.rgb_files = os.listdir(self.rgb_folder)
        self.depth_files.sort()
        self.rgb_files.sort()

        self.traj_f = os.path.join(self.dir, traj_filename)
        self.trajectory = None
        self.readTrajectory()
        depth_index_max,depth_index_min = int(self.depth_files[-1][5:10]),int(self.depth_files[0][5:10])
        rgb_index_max,rgb_index_min = int(self.rgb_files[-1][5:10]),int(self.rgb_files[0][5:10])
        tra_indexes = np.array(list(self.trajectory.keys()))
        self.index_min = max(rgb_index_min,depth_index_min,tra_indexes[0])
        self.index_max = min(rgb_index_max,depth_index_max,tra_indexes[-1])   

    def readTrajectory(self):
        self.trajectory = {}

        f = open(self.traj_f,'r')
        T_WC = []
        current_id = None
        for line in f.readlines():
            data = line.split(' ')
            if(len(data) == 3):
                if T_WC:
                    T_WC = np.array(T_WC)
                    r = Rotation.from_matrix(T_WC[:3,:3])
                    T_WC[:3,:3] = r.as_matrix()
                    self.trajectory[current_id] = np.array(T_WC).reshape(4,4)
                current_id = int(data[0])
                T_WC = []

            elif(len(data) == 4):
                T_WC.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
        f.close()

    def getDataFromIndex(self, index):
        """
        The read pose can transform points in camera frame to world frame.
        """
        # normally start from 1: image00001.png
        if(index<self.index_min or index>self.index_max):
            return None,None,None
        image_f = self.rgb_files[index]
        depth_f = self.depth_files[index]
        image_f = os.path.join(self.rgb_folder, image_f)
        depth_f = os.path.join(self.depth_folder, depth_f)

        rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        pose = self.trajectory[index]

        return rgb_img, depth_img, pose.astype(np.float32)
    
    def getPathFromIndex(self, index):
        # normally start from 1: image00001.png
        if(index<self.index_min or index>self.index_max):
            return None,None
        image_f = self.rgb_files[index]
        depth_f = self.depth_files[index]
        image_f = os.path.join(self.rgb_folder, image_f)
        depth_f = os.path.join(self.depth_folder, depth_f)

        # rgb_img = cv2.imread(image_f,cv2.IMREAD_UNCHANGED)
        # depth_img = cv2.imread(depth_f,cv2.IMREAD_UNCHANGED)
        # pose = self.trajectory[index]
        return image_f, depth_f
    
    def getPoseFromIndex(self, index):
        pose = self.trajectory[index]
        return pose

    def getCameraMatrix(self):
        K = np.array([[544.47329,0,320],[0,544.47329,240],[0,0,1]])
        return K.astype(np.float32)