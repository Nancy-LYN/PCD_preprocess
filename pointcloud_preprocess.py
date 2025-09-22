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
    import os
    # 自动将 /input/ 或 /outs/ 路径映射到 pointcloud_preprocess/input/ 和 outs/
    base_dir = os.path.dirname(__file__)
    if file_path.startswith("/input/"):
        file_path = os.path.join(base_dir, "input", os.path.basename(file_path))
    elif file_path.startswith("/outs/"):
        file_path = os.path.join(base_dir, "outs", os.path.basename(file_path))
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded point cloud: {pcd}")
    return pcd

# 2. 去噪（统计滤波）
def denoise_point_cloud(pcd, nb_neighbors=10, std_ratio=0.05):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Statistical outlier removal: {len(ind)} points remain")
    return cl

# 3. 下采样（体素网格）
def downsample_point_cloud(pcd, voxel_size=0.01):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel downsampled: {len(pcd_down.points)} points remain")
    return pcd_down

# 4. 空洞填补（Poisson重建）
def fill_holes_poisson(pcd, depth=8, density_threshold=0.1):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print("Poisson surface reconstruction done.")
    # 剔除异常面片：根据密度阈值筛选
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"Removed {np.sum(vertices_to_remove)} low-density vertices.")
    return mesh

# 网格Laplacian平滑
def smooth_mesh_laplacian(mesh, iterations=10):
    mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    print(f"Laplacian mesh smoothing done ({iterations} iterations).")
    return mesh_smooth

# 5. 密度均匀化（体素下采样）
def uniform_density(pcd, voxel_size=0.5):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel downsampled: {len(pcd_down.points)} points remain")
    return pcd_down

# 6. 保存点云/网格
def save_result(obj, out_path):
    import os
    # 自动将 /input/ 或 /outs/ 路径映射到 pointcloud_preprocess/input/ 和 outs/
    base_dir = os.path.dirname(__file__)
    if out_path.startswith("/input/"):
        out_path = os.path.join(base_dir, "input", os.path.basename(out_path))
    elif out_path.startswith("/outs/"):
        out_path = os.path.join(base_dir, "outs", os.path.basename(out_path))
    # 自动创建输出目录
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    parser.add_argument('--voxel_size', type=float, default=0.005, help='体素下采样大小')
    parser.add_argument('--mls_radius', type=float, default=5.0, help='MLS光滑搜索半径')
    parser.add_argument('--poisson_depth', type=int, default=8, help='Poisson重建深度')
    args = parser.parse_args()

    # 自动创建输出子文件夹
    import os
    out_base = os.path.dirname(args.output)
    out_name = os.path.splitext(os.path.basename(args.output))[0]
    out_dir = os.path.join(out_base, out_name)
    os.makedirs(out_dir, exist_ok=True)
    def save_to_subfolder(obj, filename):
        save_result(obj, os.path.join(out_dir, filename))

    # 步骤1：加载
    pcd = load_point_cloud(args.input)
    print(f"加载后点数: {len(pcd.points)}")
    # 步骤2：去噪
    pcd = denoise_point_cloud(pcd)
    print(f"去噪后点数: {len(pcd.points)}")
    # 保存去噪后的点云
    save_to_subfolder(pcd, f'{out_name}.denoise.xyz')
    # 步骤3：下采样
    pcd = downsample_point_cloud(pcd, voxel_size=args.voxel_size)
    print(f"下采样后点数: {len(pcd.points)}")
    # 保存下采样后的点云
    save_to_subfolder(pcd, f'{out_name}.downsample.xyz')
    # 下采样后估算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    print("下采样后已估算法线")
    # 步骤4：Poisson重建并剔除异常面片
    mesh = fill_holes_poisson(pcd, depth=args.poisson_depth, density_threshold=0.1)
    # 步骤4.1：Laplacian平滑
    mesh_smooth = smooth_mesh_laplacian(mesh, iterations=1000)
    # 步骤5：从平滑后的网格表面采样点云
    sampled_pcd_smooth = mesh_smooth.sample_points_poisson_disk(number_of_points=3000)
    print(f"平滑网格采样后点数: {len(sampled_pcd_smooth.points)}")
    # 步骤6：保存结果
    save_to_subfolder(mesh, f'{out_name}.ply')
    save_to_subfolder(mesh_smooth, f'{out_name}.smooth.ply')
    save_to_subfolder(sampled_pcd_smooth, f'{out_name}.smooth.xyz')

    # 多轮迭代：去噪-泊松重建-平滑-采样
    iter_num = 2  # 可调整迭代次数
    pcd_iter = sampled_pcd_smooth
    for i in range(iter_num):
        print(f"\n--- 迭代 {i+1} ---")
        pcd_iter = denoise_point_cloud(pcd_iter)
        print(f"迭代{i+1}去噪后点数: {len(pcd_iter.points)}")
        pcd_iter.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        mesh_iter = fill_holes_poisson(pcd_iter, depth=args.poisson_depth, density_threshold=0.1)
        mesh_iter_smooth = smooth_mesh_laplacian(mesh_iter, iterations=1000)
        pcd_iter_sampled = mesh_iter_smooth.sample_points_poisson_disk(number_of_points=3000)
        print(f"迭代{i+1}平滑网格采样后点数: {len(pcd_iter_sampled.points)}")
        save_to_subfolder(mesh_iter, f'{out_name}.iter{i+1}.ply')
        save_to_subfolder(mesh_iter_smooth, f'{out_name}.iter{i+1}.smooth.ply')
        save_to_subfolder(pcd_iter_sampled, f'{out_name}.iter{i+1}.smooth.xyz')
        pcd_iter = pcd_iter_sampled
