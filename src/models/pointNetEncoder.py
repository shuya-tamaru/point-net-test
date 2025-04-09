import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import get_activation, TNet


class PointNetEncoder(nn.Module):
    def __init__(self, channel=3, activation='relu'):
        super(PointNetEncoder, self).__init__()
        self.activation = get_activation(activation)

        # 入力変換ネットワーク (3D座標の変換)
        self.t_net3 = TNet(k=channel)

        # 初期の特徴抽出
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # 特徴変換ネットワーク (特徴空間での変換)
        self.t_net64 = TNet(k=64)

        # さらなる特徴抽出
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # バッチ正規化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """順伝播"""
        n_pts = x.size()[2]

        # 入力変換 (3D座標の位置合わせ)
        trans1 = self.t_net3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)  # バッチ行列積
        x = x.transpose(2, 1)

        # 特徴抽出
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))

        # 特徴変換 (特徴空間での位置合わせ)
        trans2 = self.t_net64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2, 1)

        # 点ごとの特徴を保存
        point_feat = x

        # さらなる特徴抽出
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))

        # グローバルな特徴を取得（最大値プーリング）
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # グローバル特徴、変換行列、点ごとの特徴を返す
        return x, trans1, trans2, point_feat
