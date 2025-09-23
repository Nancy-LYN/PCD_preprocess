import open3d as o3d
import argparse
import os

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved denoised point cloud to {out_path}")

def denoise_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Original points: {len(pcd.points)}, Denoised points: {len(ind)}")
    return pcd.select_by_index(ind)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云去噪（统计离群点移除）")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件路径')
    parser.add_argument('--nb_neighbors', type=int, default=20, help='统计邻域点数')
    parser.add_argument('--std_ratio', type=float, default=2.0, help='标准差倍数阈值')
    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    pcd_denoised = denoise_point_cloud(pcd, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    save_point_cloud(pcd_denoised, args.output)

# 用法示例：
# python pointcloud_denoise.py --input input/xxx.xyz --output outs/xxx_denoised.xyz --nb_neighbors 20 --std_ratio 2.0
