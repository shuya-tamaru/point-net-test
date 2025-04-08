import open3d as o3d


def visualize_point_cloud(pcd):
    print("Visualizing point cloud...")
    o3d.visualization.draw_geometries([pcd])
