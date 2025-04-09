import os
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
from .point_cloud_dataset import PointCloudDataset
from .point_net import PointNet


def learning(data_dir, num_classes):
    train_dataset = PointCloudDataset(data_dir, num_classes)

    batch_size = 4
    shuffle = True
    num_workers = 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f"Number of batches in train_loader: {len(train_loader)}")

    model = PointNet(num_classes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            data = data.to(device)
            print(data)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)  # 損失を計算
            loss.backward()  # 逆伝播（勾配の計算）
            optimizer.step()  # パラメータの更新

            running_loss += loss.item()
            if i % 10 == 9:  # 10 バッチごとに損失を出力
                print(
                    f"[{epoch + 1}, {i + 1}/{len(train_loader)}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        print(f"Epoch {epoch + 1} finished.")

    print("Finished Training")
    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pointnet_segmentation.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return
