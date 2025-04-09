import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import get_activation, TNet
from .pointNetEncoder import PointNetEncoder


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=4, channel=3, activation='relu'):
        super(PointNetSeg, self).__init__()
        self.num_classes = num_classes
        self.activation = get_activation(activation)

        # エンコーダー
        self.encoder = PointNetEncoder(channel=channel, activation=activation)

        # セグメンテーション用のデコーダー
        # 1088 = 64(点特徴) + 1024(グローバル特徴)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # 変換行列の正則化損失
        self.transform_regularizer = nn.MSELoss()

    def forward(self, x):
        """順伝播"""
        # エンコーダーで特徴抽出
        global_feat, trans1, trans2, point_feat = self.encoder(x)

        # グローバル特徴を各点に拡張して結合
        n_pts = x.size()[2]
        global_feat_expanded = global_feat.view(-1,
                                                1024, 1).repeat(1, 1, n_pts)

        # グローバル特徴と点ごとの特徴を結合
        concat_feat = torch.cat([point_feat, global_feat_expanded], 1)

        # セグメンテーションのための特徴処理
        x = self.activation(self.bn1(self.conv1(concat_feat)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        # 出力の形を調整 (B, C, N) -> (B, N, C)
        x = x.transpose(2, 1)
        return x, trans1, trans2

    def get_transform_loss(self, trans):
        """変換行列の正則化損失を計算"""
        # 直交行列に近づける正則化（単位行列との差をMSEで計算）
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = self.transform_regularizer(
            torch.bmm(trans, trans.transpose(2, 1)), I.repeat(trans.size()[0], 1, 1))
        return loss
