import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'elu':
        return nn.ELU()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    else:
        return nn.ReLU()


class TNet(nn.Module):

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """順伝播"""
        batch_size = x.size()[0]

        # 1D畳み込みで特徴抽出
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # グローバルな特徴を取得（最大値プーリング）
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # 全結合層で変換行列を生成
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 単位行列に初期化（恒等変換からの変化を学習する）
        iden = torch.eye(self.k).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        # 変換行列の形に変形して単位行列を加える
        x = x.view(-1, self.k, self.k) + iden
        return x
