import open3d as o3d
import numpy as np


def visualize_segmentation(points, labels):
    colors = np.zeros((len(points), 3))

    class_colors = [
        [0, 0, 1],    # 床: 青
        [0, 1, 0],    # 壁: 緑
        [1, 0, 0],    # 天井: 赤
        [1, 1, 0],    # 家具: 黄色
        [1, 0, 1],    # その他の家具: マゼンタ
        [0, 1, 1],    # ドア/窓: シアン
        [0.5, 0.5, 0.5]  # その他: グレー
    ]

    for i in range(len(points)):
        label = int(labels[i])
        if label < len(class_colors):
            colors[i] = class_colors[label]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Segmentation Result")
