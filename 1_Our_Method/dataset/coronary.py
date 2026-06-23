import sys

sys.path.append('.')
import os
import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


class Coronary(data.Dataset):
    """
    Dataset class for loading coronary artery point cloud data.
    This dataset loads paired partial and complete point clouds from disk.
    It supports three dataset splits: train, val, and test.
    Args:
        dataroot (str): Root directory of the dataset.
        split (str): Dataset split, must be one of ['train', 'val', 'test'].
        category (str): Category name or identifier. Reserved for category-specific filtering.
    """

    def __init__(self, dataroot, split, category):
        assert split in ['train', 'val', 'test'], "split error value!"

        self.dataroot = dataroot
        self.split = split
        self.category = category

        self.partial_paths, self.complete_paths = self._load_data()

    def __getitem__(self, index):
        """
        Get a pair of partial and complete point clouds by index.
        Args:
            index (int): Index of the sample.
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                A tuple containing:
                - partial point cloud tensor with shape (N, 3)
                - complete point cloud tensor with shape (M, 3)
        """
        partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]

        partial_pc = self.read_point_cloud(partial_path)
        complete_pc = self.read_point_cloud(complete_path)

        # if partial_pc.shape[0] > 4096:
        #     partial_pc = self.random_sample(partial_pc, 4096)
        # elif partial_pc.shape[0] < 4096:
        #     partial_pc = np.concatenate([partial_pc, self.random_sample(partial_pc, 4096-partial_pc.shape[0])])

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of complete point cloud samples.
        """
        return len(self.complete_paths)

    def _load_data(self):
        """
        Load partial and complete point cloud file paths from the split list file.
        The split file should be named as:
            train.list, val.list, or test.list
        Each line in the list file should follow the format:
            category/model_id
        Returns:
            tuple(list, list):
                - partial_paths: List of partial point cloud file paths.
                - complete_paths: List of complete point cloud file paths.
        """
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            complete_paths.append(os.path.join(self.dataroot, self.split, 'complete', category + '.ply'))

        return partial_paths, complete_paths

    def read_point_cloud(self, path):
        """
        Read a point cloud file and convert it to a NumPy array.
        Args:
            path (str): Path to the point cloud file.
        Returns:
            np.ndarray: Point cloud coordinates with shape (N, 3), dtype float32.
        """
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)

    def random_sample(self, pc, n):
        """
        Randomly sample a fixed number of points from a point cloud.
        If the point cloud contains fewer than n points, points will be randomly
        duplicated to reach the target number.
        Args:
            pc (np.ndarray): Input point cloud with shape (N, 3).
            n (int): Number of points to sample.
        Returns:
            np.ndarray: Sampled point cloud with shape (n, 3).
        """
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
        return pc[idx[:n]]
