import numpy as np
import open3d as o3d


def label_point_cloud(point_cloud, verbose=True):
    points = np.asarray(point_cloud.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min

    floor_threshold = z_min + 0.1 * z_range
    ceiling_threshold = z_max - 0.1 * z_range

    labels = np.zeros(len(points), dtype=np.int32)

    labels[points[:, 2] < floor_threshold] = 0  # 床
    labels[points[:, 2] > ceiling_threshold] = 2  # 天井
    labels[(points[:, 2] >= floor_threshold) & (
        points[:, 2] <= ceiling_threshold)] = 1  # 壁
    labels[(points[:, 2] > floor_threshold) & (
        points[:, 2] < ceiling_threshold)] = 3  # 家具
    print(f"labels: {labels}")
    if verbose:
        print(f"点群のラベル分布:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            label_names = {0: 'その他', 1: '床', 2: '壁', 3: '天井'}
            print(
                f"{label_names[label]}: {count}点 ({count/len(labels)*100:.2f}%)")

    return labels
