"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *


curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=20, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, xyz, get_flatten_curve_idxs=False):
        flatten_curve_idxs = {}
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points, flatten_curve_idxs_11 = self.cic11(xyz, l0_points)
        flatten_curve_idxs['flatten_curve_idxs_11'] = flatten_curve_idxs_11
        l1_xyz, l1_points, flatten_curve_idxs_12 = self.cic12(l1_xyz, l1_points)
        flatten_curve_idxs['flatten_curve_idxs_12'] = flatten_curve_idxs_12

        l2_xyz, l2_points, flatten_curve_idxs_21 = self.cic21(l1_xyz, l1_points)
        flatten_curve_idxs['flatten_curve_idxs_21'] = flatten_curve_idxs_21
        l2_xyz, l2_points, flatten_curve_idxs_22 = self.cic22(l2_xyz, l2_points)
        flatten_curve_idxs['flatten_curve_idxs_22'] = flatten_curve_idxs_22

        l3_xyz, l3_points, flatten_curve_idxs_31 = self.cic31(l2_xyz, l2_points)
        flatten_curve_idxs['flatten_curve_idxs_31'] = flatten_curve_idxs_31
        l3_xyz, l3_points, flatten_curve_idxs_32 = self.cic32(l3_xyz, l3_points)
        flatten_curve_idxs['flatten_curve_idxs_32'] = flatten_curve_idxs_32
 
        l4_xyz, l4_points, flatten_curve_idxs_41 = self.cic41(l3_xyz, l3_points)
        flatten_curve_idxs['flatten_curve_idxs_41'] = flatten_curve_idxs_41
        l4_xyz, l4_points, flatten_curve_idxs_42 = self.cic42(l4_xyz, l4_points)
        flatten_curve_idxs['flatten_curve_idxs_42'] = flatten_curve_idxs_42

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        if get_flatten_curve_idxs:
            return x, flatten_curve_idxs
        else:
            return x
