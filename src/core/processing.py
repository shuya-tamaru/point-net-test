import numpy as np
import open3d as o3d
import torch
import os


def preprocess_point_cloud(pcd, num_points=4096):
    print(f"[INFO] Preprocessing point cloud...")
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"[SUCCESS] Preprocessing completed")

    return pcd, points
