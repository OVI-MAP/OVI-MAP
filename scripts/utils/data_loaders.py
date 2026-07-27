import os, copy, glob, json
from os.path import join as pjoin
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
