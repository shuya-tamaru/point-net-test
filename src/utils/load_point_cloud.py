import open3d as o3d


def load_point_cloud(file_path):
    print(f"Loading point cloud from {file_path}...")

    pcd = o3d.io.read_point_cloud(file_path)

    print(f"Loaded point cloud with {len(pcd.points)} points")
    return pcd
