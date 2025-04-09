import torch
import numpy as np
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ..learning.point_net import PointNet


def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model to the appropriate device
    model = PointNet(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, device


def load_ply(file_path):
    """Load a PLY file and return the point cloud as a numpy array."""
    ply_data = PlyData.read(file_path)
    points = np.vstack([ply_data['vertex'][axis]
                       for axis in ('x', 'y', 'z')]).T
    return points


def segment_ply(model, points, device):
    data = Data(
        pos=torch.tensor(points, dtype=torch.float32),
        batch=torch.zeros(len(points), dtype=torch.long)
    )
    data = data.to(device)

    with torch.no_grad():
        predictions = model(data)  # Data 型をモデルに渡す
    return predictions.argmax(dim=1).cpu().numpy()


def segment_ply_with_batches(model, points, device, batch_size=100000):
    dataset = [
        Data(
            pos=torch.tensor(points[i:i+batch_size], dtype=torch.float32),
            batch=torch.zeros(
                min(batch_size, len(points) - i), dtype=torch.long)
        )
        for i in range(0, len(points), batch_size)
    ]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)  # データを GPU に移動
            predictions = model(data)  # モデルにデータを渡して推論
            all_predictions.append(predictions.argmax(
                dim=1).cpu().numpy())  # 結果をリストに追加

    return np.concatenate(all_predictions)


def save_segmented_ply(input_ply_path, output_ply_path, labels):
    """Save the segmented PLY file with labels."""
    ply_data = PlyData.read(input_ply_path)
    vertex_data = ply_data['vertex'].data
    new_vertex_data = np.array(
        [tuple(list(vertex) + [label])
         for vertex, label in zip(vertex_data, labels)],
        dtype=vertex_data.dtype.descr + [('label', 'u1')]
    )

    new_ply = PlyData([PlyElement.describe(
        new_vertex_data, 'vertex')], text=True)
    new_ply.write(output_ply_path)


def segment(input_ply_path, output_ply_path, model_path, num_classes):
    model, device = load_model(model_path, num_classes)
    points = load_ply(input_ply_path)
    labels = segment_ply_with_batches(model, points, device)
    save_segmented_ply(input_ply_path, output_ply_path, labels)
