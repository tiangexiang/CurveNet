"""
@Author: Vinit Sarode
@Contact: vinitsarode5@gmail.com
@File: visualize_curves.py
@Time: 2025/03/03 11:17 AM
"""

import argparse
import torch
import torch.nn as nn
from data import ModelNet40
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from models.curvenet_cls import CurveNet

def visualize_point_cloud(pcd, curves, axis=False, title=""):
    x, y, z= pcd[..., 0], pcd[..., 1], pcd[..., 2]
    fig = go.Figure(
        layout=dict(
            scene=dict(
                xaxis=dict(visible=axis),
                yaxis=dict(visible=axis),
                zaxis=dict(visible=axis)
            ),
            title=title,
            title_x=0.5
        )
    )
    fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, 
                mode='markers',
                marker=dict(size=1)
            ))

    for curve in curves:
        x, y, z= curve[..., 0], curve[..., 1], curve[..., 2]
        fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z, 
                    # mode='markers',
                    marker=dict(size=1),
                    line=dict(
                        color='darkred',
                        width=2
                    )
                ))
    fig.show()

def visualize(args):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=1, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = CurveNet().to(device)
    weights = torch.load(args.model_path, map_location='cpu')
    weights = {k[7:]: v for k, v in weights.items()}
    model.load_state_dict(weights)

    model = model.eval()
    for idx, (data, label) in enumerate(test_loader):
        if idx >= args.no_of_samples:
            break

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits, flatten_cur = model(data, get_flatten_curve_idxs=True)
        data = data.permute(0, 2, 1).detach().cpu().numpy()[0]

        curves_dict = {}
        for key, val in flatten_cur.items():
            if val is not None:
                curves = []
                val_np = val.cpu().detach().numpy()
                for idx in range(val_np.shape[0]):
                    curves.append(data[val_np[idx]])
                curves_dict[key] = curves
            
        visualize_point_cloud(data, curves_dict[args.visualize_curve], title=args.visualize_curve)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visualize_curve', type=str, default='flatten_curve_idxs_11', 
                        help='Choose which curve to visualize based on model architecture',
                        choices=['flatten_curve_idxs_11', 'flatten_curve_idxs_12',
                                 'flatten_curve_idxs_21', 'flatten_curve_idxs_22'])
    parser.add_argument('--no_of_samples', type=int, default=3, 
                        help='No of point clouds to visualize with curves')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    visualize(args=args)