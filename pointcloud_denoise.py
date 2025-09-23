import open3d as o3d
import argparse
import os
import numpy as np

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved denoised point cloud to {out_path}")


def denoise_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, normal_angle=45):
    # 1. Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    print(f"After statistical outlier removal: {len(pcd.points)} points")

    # 2. Remove sharp spikes by normal deviation
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    keep_indices = []
    angle_threshold = np.deg2rad(normal_angle)
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(points[i], nb_neighbors)
        neighbor_normals = normals[idx]
        avg_normal = np.mean(neighbor_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal) + 1e-8
        cos_angle = np.dot(normals[i], avg_normal) / (np.linalg.norm(normals[i]) * np.linalg.norm(avg_normal) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            keep_indices.append(i)
    print(f"After normal deviation removal: {len(keep_indices)} points")
    return pcd.select_by_index(keep_indices)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="点云去噪（统计离群点+法线偏差去除尖锐突起）")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件路径')
    parser.add_argument('--nb_neighbors', type=int, default=20, help='统计邻域点数')
    parser.add_argument('--std_ratio', type=float, default=2.0, help='标准差倍数阈值')
    parser.add_argument('--normal_angle', type=float, default=45, help='法线偏差阈值（度），大于该夹角的点将被去除')
    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    pcd_denoised = denoise_point_cloud(pcd, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio, normal_angle=args.normal_angle)
    save_point_cloud(pcd_denoised, args.output)

# 用法示例：
# python pointcloud_denoise.py --input input/xxx.xyz --output outs/xxx_denoised.xyz --nb_neighbors 20 --std_ratio 2.0 --normal_angle 45
