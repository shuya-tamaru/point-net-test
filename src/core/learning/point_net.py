import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_max_pool


class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv4 = nn.Conv1d(1088, 512, 1)  # 1088 = 1024 + 64 (グローバル + 局所)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        x = data.pos  # [N, 3]
        batch = data.batch

        # 点の座標を変換 [N, 3] -> [1, 3, N]
        x = x.transpose(1, 0).unsqueeze(0)

        # 特徴抽出
        local_feat = F.relu(self.bn1(self.conv1(x)))  # [1, 64, N]
        x = F.relu(self.bn2(self.conv2(local_feat)))  # [1, 128, N]
        x = self.bn3(self.conv3(x))  # [1, 1024, N]

        # グローバル特徴の抽出
        # まず形状変換 [1, 1024, N] -> [N, 1024]
        point_feat = x.squeeze(0).transpose(0, 1)  # [N, 1024]

        # バッチごとにプール（グローバル特徴）
        # 結果: [バッチ数, 1024]
        global_feat = global_max_pool(point_feat, batch)  # [バッチ数, 1024]

        expanded_global_feat = global_feat[batch]  # [N, 512]

        # 局所特徴とグローバル特徴を結合
        local_feat = local_feat.squeeze(0).transpose(0, 1)  # [N, 64]
        concat_feat = torch.cat(
            [local_feat, expanded_global_feat], dim=1)  # [N, 64+512=576]

        # セグメンテーション用デコーダー
        # 結合特徴を戻す [N, 576] -> [1, 576, N]
        concat_feat = concat_feat.transpose(1, 0).unsqueeze(0)

        # 点ごとの分類
        x = F.relu(self.bn5(self.conv4(concat_feat)))
        x = F.relu(self.bn6(self.conv5(x)))
        x = F.relu(self.bn7(self.conv6(x)))
        x = self.dropout(x)
        x = self.conv7(x)  # [1, num_classes, N]

        # 形状変換 [1, num_classes, N] -> [N, num_classes]
        x = x.squeeze(0).transpose(0, 1)

        return F.log_softmax(x, dim=-1)
