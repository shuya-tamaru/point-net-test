from plyfile import PlyData
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data


class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_classes, transform=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.file_list = [f for f in os.listdir(
            data_dir) if f.endswith('.ply')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        points, labels = self.read_ply(file_path)
        labels_contiguous = np.copy(labels)
        data = Data(pos=torch.from_numpy(points).float(),
                    y=torch.from_numpy(labels_contiguous).long())

        if self.transform:
            data = self.transform(data)

        return data

    def read_ply(self, file_path):
        ply = PlyData.read(file_path)
        vertices = ply['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        labels = vertices['scalar_label']
        return points, labels

    def normalize_points(self, points):
        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
        points = points / furthest_distance
        return points
