import open3d as o3d
import numpy as np
import argparse
import os
from scipy.ndimage import binary_dilation

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved dilated point cloud to {out_path}")

def voxel_dilate_point_cloud(pcd, voxel_size=0.01, dilation_iter=2):
    points = np.asarray(pcd.points)
    # 1. 映射到体素网格
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 3
    grid = np.zeros(grid_shape, dtype=bool)
    indices = np.floor((points - min_bound) / voxel_size).astype(int) + 1
    grid[indices[:,0], indices[:,1], indices[:,2]] = True
    # 2. 体素膨胀
    grid_dilated = binary_dilation(grid, iterations=dilation_iter)
    # 3. 体素中心转回点云
    dilated_indices = np.argwhere(grid_dilated)
    dilated_points = dilated_indices - 1
    dilated_points = dilated_points * voxel_size + min_bound + voxel_size / 2
    pcd_dilated = o3d.geometry.PointCloud()
    pcd_dilated.points = o3d.utility.Vector3dVector(dilated_points)
    return pcd_dilated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云体素膨胀平滑")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径（.xyz）')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件路径（.xyz）')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='体素大小')
    parser.add_argument('--dilation_iter', type=int, default=2, help='膨胀迭代次数')
    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    print(f"Loaded point cloud: {args.input}, 点数: {len(pcd.points)}")
    pcd_dilated = voxel_dilate_point_cloud(pcd, voxel_size=args.voxel_size, dilation_iter=args.dilation_iter)
    print(f"Dilated point cloud: {len(pcd_dilated.points)} points")
    save_point_cloud(pcd_dilated, args.output)

# 用法示例：
# python pointcloud_voxel_dilate.py --input input/xxx.xyz --output outs/xxx_dilated.xyz --voxel_size 0.01 --dilation_iter 2
