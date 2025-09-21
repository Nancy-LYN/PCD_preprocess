"""
pointcloud_preprocess.py

点云预处理：去噪、光滑、空洞填补、密度均匀化
依赖：open3d
安装：pip install open3d
"""
import open3d as o3d
import numpy as np
import argparse

# 1. 加载点云
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded point cloud: {pcd}")
    return pcd

# 2. 去噪（统计滤波）
def denoise_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Statistical outlier removal: {len(ind)} points remain")
    return cl

# 3. 表面光滑（MLS）
def smooth_point_cloud(pcd, search_radius=1.0):
    pcd_smoothed = pcd.voxel_down_sample(voxel_size=0.5)
    pcd_smoothed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    print("Normals estimated for MLS smoothing.")
    return pcd_smoothed

# 4. 空洞填补（Poisson重建）
def fill_holes_poisson(pcd, depth=8):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print("Poisson surface reconstruction done.")
    return mesh

# 5. 密度均匀化（体素下采样）
def uniform_density(pcd, voxel_size=0.5):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel downsampled: {len(pcd_down.points)} points remain")
    return pcd_down

# 6. 保存点云/网格
def save_result(obj, out_path):
    if isinstance(obj, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(out_path, obj)
        print(f"Saved point cloud to {out_path}")
    elif isinstance(obj, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(out_path, obj)
        print(f"Saved mesh to {out_path}")
    else:
        print("Unknown object type for saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云预处理：去噪、光滑、空洞填补、密度均匀化")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='体素下采样大小')
    parser.add_argument('--mls_radius', type=float, default=1.0, help='MLS光滑搜索半径')
    parser.add_argument('--poisson_depth', type=int, default=8, help='Poisson重建深度')
    args = parser.parse_args()

    # 步骤1：加载
    pcd = load_point_cloud(args.input)
    # 步骤2：去噪
    pcd = denoise_point_cloud(pcd)
    # 步骤3：光滑
    pcd = smooth_point_cloud(pcd, search_radius=args.mls_radius)
    # 步骤4：空洞填补（重建网格）
    mesh = fill_holes_poisson(pcd, depth=args.poisson_depth)
    # 步骤5：密度均匀化（可选，对点云）
    pcd_uniform = uniform_density(pcd, voxel_size=args.voxel_size)
    # 步骤6：保存结果
    save_result(mesh, args.output)
    # 如需保存点云而非网格，可用：save_result(pcd_uniform, args.output.replace('.ply', '_pcd.ply'))
