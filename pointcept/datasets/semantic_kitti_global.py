import os
import numpy as np
import pyvista
import pickle
import open3d as o3d
from tqdm import tqdm

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SemanticKITTIGlobalDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)
        # 缓存文件路径：放在 data_root 下面，名字固定
        self.cache_path = os.path.join(self.data_root,
                                       f'standardized_{self.split}.pkl')

        # 第一次初始化时建立/加载映射表：主帧 -> 最优互补帧
        self.best_pairs = {}
        self._prepare_best_pairs()

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, seq, "velodyne")
            seq_files = sorted(os.listdir(seq_folder))
            data_list += [
                os.path.join(seq_folder, file) for file in seq_files
            ]
        return data_list

    # ------------ 1. 评价函数 ------------ #
    def similarity_score(self, fitness):
        """
        将 fitness ∈ [0,1] 映射到“适中相似度”得分 ∈ [0,1]，
        峰值落在 fitness ≈ 0.5
        二次函数: -4(x-0.5)^2 + 1
        """
        return -(fitness - 0.7) ** 2 + 1.0

    # ------------ 2. 改进的 ICP 打分 ------------ #
    def _icp_score(self, src_path, dst_path, init_pose):
        """
        init_pose: 4x4 初始位姿，由 get_pose 提供
        返回 fitness ∈ [0,1]
        """
        src_pc = np.fromfile(src_path, np.float32).reshape(-1, 4)[:, :3]
        dst_pc = np.fromfile(dst_path, np.float32).reshape(-1, 4)[:, :3]

        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pc))
        dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_pc))

        src = src.voxel_down_sample(0.3)
        dst = dst.voxel_down_sample(0.3)

        src.estimate_normals()
        dst.estimate_normals()

        reg = o3d.pipelines.registration.registration_icp(
            src, dst, max_correspondence_distance=1.0,
            init=init_pose,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

        return reg.fitness

    # ------------ 3. 改进的 _prepare_best_pairs ------------ #
    def _prepare_best_pairs(self):
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.best_pairs = pickle.load(f)
            new_best_pairs = {}
            for a, b in self.best_pairs.items():
                parts = a.split('/')
                seq_part = parts[-2]
                file_part = parts[-1]
                new_a = os.path.join(self.data_root, seq_part, "velodyne", file_part)

                parts = b.split('/')
                seq_part = parts[-2]
                file_part = parts[-1]
                new_b = os.path.join(self.data_root, seq_part, "velodyne", file_part)
                # 将修改后的键值对添加到新字典中
                new_best_pairs[new_a] = new_b

            # 更新原字典
            self.best_pairs = new_best_pairs
            return

        print(f'[ICP] building pair cache for split={self.split} ...')
        data_list = self.get_data_list()

        seq2frames = {}
        for path in data_list:
            seq = path.split(os.sep)[-2]
            seq2frames.setdefault(seq, []).append(path)

        for seq, frames in seq2frames.items():
            frames = sorted(frames)
            seq_len = len(frames)

            # 预加载所有位姿，避免重复 IO
            poses = {f: self.get_pose(f) for f in frames}

            for i, main_path in enumerate(tqdm(frames, desc=f'ICP {seq}')):
                best_score = -np.inf
                best_path = None

                # 时间窗口：±3 秒（KITTI 10Hz，30 帧）
                max_delta = 10  # 30 帧
                left = max(0, i - max_delta)
                right = min(seq_len, i + max_delta + 1)

                for j in range(left, right):
                    if j == i:
                        continue

                    cand_path = frames[j]

                    # 1. 计算初始位姿：src -> dst 的相对位姿
                    pose_i = poses[main_path]
                    pose_j = poses[cand_path]
                    init_pose = np.linalg.inv(pose_i) @ pose_j  # 4x4

                    # 2. ICP
                    try:
                        fitness = self._icp_score(main_path, cand_path, init_pose)
                    except Exception as e:
                        fitness = 0.0

                    # 3. 评价
                    score = self.similarity_score(fitness)
                    if score > best_score:
                        best_score = score
                        best_path = cand_path

                # 兜底：防止全部失败
                if best_path is None:
                    best_path = frames[max(0, i - 1)] if i > 0 else frames[min(seq_len - 1, i + 1)]

                self.best_pairs[main_path] = best_path

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.best_pairs, f)
        print(f'[ICP] cache saved to {self.cache_path}')

    # 重写互补帧获取逻辑
    def get_complementary_frame(self, data_path):
        """
        现在直接查表即可
        """
        return self.best_pairs.get(data_path, None)


    def get_complementary_frame_old(self, data_path):
        # 解析当前帧的文件名和路径
        dir_path, file_name = os.path.split(data_path)
        frame_num = file_name.split('.')[0]

        # 计算目标帧的编号
        target_frame_prev = f"{int(frame_num) - 1:06d}.bin"
        target_frame_next = f"{int(frame_num) + 1:06d}.bin"

        # 构造目标帧的路径
        target_path_prev = os.path.join(dir_path, target_frame_prev)
        target_path_next = os.path.join(dir_path, target_frame_next)

        # 检查目标帧是否存在，优先选择前 1 帧，若不存在则选择后 1 帧
        if os.path.exists(target_path_prev):
            return target_path_prev
        elif os.path.exists(target_path_next):
            return target_path_next
        else:
            # 如果前后都不存在合适的帧，可以选择相邻的前一帧或后一帧，或者返回 None 表示没有找到有效的帧
            return None

    def parse_calibration(self, filename):
        """读取校准文件，返回4x4的变换矩阵"""
        calib = {}
        with open(filename) as calib_file:
            for line in calib_file:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]
                pose = np.zeros((4, 4))
                pose[0, :4] = values[0:4]
                pose[1, :4] = values[4:8]
                pose[2, :4] = values[8:12]
                pose[3, 3] = 1.0
                calib[key] = pose
        return calib

    def parse_poses(self, filename, calibration):
        """读取姿态文件，返回变换后的姿态列表"""
        poses = []
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        with open(filename) as file:
            for line in file:
                values = [float(v) for v in line.strip().split()]
                pose = np.zeros((4, 4))
                pose[0, :4] = values[0:4]
                pose[1, :4] = values[4:8]
                pose[2, :4] = values[8:12]
                pose[3, 3] = 1.0
                transformed_pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
                poses.append(transformed_pose)
        return poses

    def j_to_i_project(self, point_j, pose_j, pose_i):
        """将点 j 从姿态 j 转换到姿态 i"""
        diff_pose = np.matmul(np.linalg.inv(pose_i), pose_j)
        pc_j = self.rigid_translate(point_j, diff_pose)
        return pc_j

    def rigid_translate(self, pc_input, extrinsic):
        """应用刚体变换"""
        pc = np.hstack((pc_input[:, :3], np.ones_like(pc_input[:, 0]).reshape(-1, 1)))
        pc = np.matmul(extrinsic, pc.T).T
        pcl = np.hstack((pc[:, :3], pc_input[:, 3:]))
        return pcl

    def get_pose(self, data_path):
        pc_filename = os.path.basename(data_path)
        # 提取索引（去掉扩展名并转换为整数）
        idx = int(pc_filename[:-4])

        pose_dir = data_path.rsplit(os.path.sep, 2)[0]
        pose_path = os.path.join(pose_dir, "poses.txt")

        calib_dir = data_path.rsplit(os.path.sep, 2)[0]
        calib_path = os.path.join(calib_dir, "calib.txt")

        calib = self.parse_calibration(calib_path)
        poses = self.parse_poses(pose_path, calib)
        # 获取对应索引的姿态数据
        pose = poses[idx]
        return pose

    # pyvista可视化
    def plot_point_cloud(self, coord, com_coord):
        # 创建点云数据
        points = pyvista.PolyData(coord)
        com_points = pyvista.PolyData(com_coord)

        plotter = pyvista.Plotter(shape=(1, 2), border=True)
        plotter.subplot(0, 0)
        plotter.add_mesh(points, render_points_as_spheres=False, point_size=2, color="blue")

        plotter.subplot(0, 1)
        plotter.add_mesh(points, render_points_as_spheres=False, point_size=2, color="blue")
        plotter.add_mesh(com_points, render_points_as_spheres=False, point_size=2, color="red")

        plotter.link_views()
        plotter.show()

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        pose = self.get_pose(data_path)
        com_data_path = self.get_complementary_frame(data_path)
        com_pose = self.get_pose(com_data_path)
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        with open(com_data_path, "rb") as b:
            com_scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        com_coord = com_scan[:, :3]
        com_strength = com_scan[:, -1].reshape([-1, 1])
        com_coord = self.j_to_i_project(com_coord, com_pose, pose)

        coord_all = np.concatenate([coord, com_coord], axis=0)  # (N+M,3)
        strength_all = np.concatenate([strength, com_strength], axis=0)  # (N+M,1)
        com_mask = np.concatenate(
            [np.zeros(len(coord), dtype=np.int64),
             np.ones(len(com_coord), dtype=np.int64)]
        )
        # self.plot_point_cloud(coord, stacked_coord)
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        com_label_file = com_data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(com_label_file):
            with open(com_label_file, "rb") as a:
                com_segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                com_segment = np.vectorize(self.learning_map.__getitem__)(
                    com_segment & 0xFFFF
                ).astype(np.int32)
        else:
            com_segment = np.zeros(com_scan.shape[0]).astype(np.int32)

        segment_all = np.concatenate([segment, com_segment], axis=0)  # (N+M,3)
        data_dict = dict(
            coord_all=coord_all,
            strength_all=strength_all,
            com_mask=com_mask,
            mask_all=com_mask,
            segment_all=segment_all,
            name=self.get_data_name(idx),
            com_name=self.get_data_name_frompath(com_data_path)
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    def get_data_name_frompath(self, file_path):
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
