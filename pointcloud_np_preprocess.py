"""
pointcloud_np_preprocess.py

点云去噪与均匀化（仅用 numpy/scipy/pyntcloud，无需 libGL）
依赖：numpy, scipy, pyntcloud
安装：pip install numpy scipy pyntcloud
"""
import numpy as np
from pyntcloud import PyntCloud
import argparse
import os
from scipy.spatial import cKDTree

# 1. 加载点云（xyz格式）
def load_xyz(file_path):
    # 自动将 /input/ 或 /outs/ 路径映射到当前脚本目录下
    if file_path.startswith("/input/"):
        file_path = os.path.join(os.path.dirname(__file__), "input", os.path.basename(file_path))
    elif file_path.startswith("/outs/"):
        file_path = os.path.join(os.path.dirname(__file__), "outs", os.path.basename(file_path))
    points = np.loadtxt(file_path)
    print(f"Loaded {points.shape[0]} points from {file_path}")
    return points

# 2. 去噪：统计滤波（移除离群点）
def statistical_outlier_removal(points, nb_neighbors=5, std_ratio=1.0):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors)
    mean_dists = np.mean(dists, axis=1)
    mean = np.mean(mean_dists)
    std = np.std(mean_dists)
    mask = mean_dists < mean + std_ratio * std
    print(f"Statistical outlier removal: {np.sum(mask)} points remain")
    return points[mask]

# 3. 均匀化：体素网格下采样
def voxel_downsample(points, voxel_size=0.5):
    coords = np.floor(points / voxel_size)
    _, idx = np.unique(coords, axis=0, return_index=True)
    down_points = points[idx]
    print(f"Voxel downsampled: {down_points.shape[0]} points remain")
    return down_points

# 4. 保存点云（xyz格式）
def save_xyz(points, out_path):
    # 自动将 /input/ 或 /outs/ 路径映射到当前脚本目录下
    if out_path.startswith("/input/"):
        out_path = os.path.join(os.path.dirname(__file__), "input", os.path.basename(out_path))
    elif out_path.startswith("/outs/"):
        out_path = os.path.join(os.path.dirname(__file__), "outs", os.path.basename(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, points, fmt='%.6f')
    print(f"Saved point cloud to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云去噪与均匀化（numpy/scipy/pyntcloud）")
    parser.add_argument('--input', type=str, default="input/test6.xyz", help='输入点云文件路径，默认 input/test6.xyz')
    parser.add_argument('--output', type=str, default="outs/test6.xyz", help='输出文件路径，默认 outs/test6.xyz')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='体素下采样大小')
    parser.add_argument('--nb_neighbors', type=int, default=10, help='统计滤波邻居数')
    parser.add_argument('--std_ratio', type=float, default=5.0, help='统计滤波标准差倍数')
    args = parser.parse_args()

    points = load_xyz(args.input)
    print(f"加载后点数: {points.shape[0]}")
    points = statistical_outlier_removal(points, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    print(f"去噪后点数: {points.shape[0]}")
    points = voxel_downsample(points, voxel_size=args.voxel_size)
    print(f"下采样后点数: {points.shape[0]}")
    save_xyz(points, args.output)
