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
def denoise_point_cloud(pcd, nb_neighbors=5, std_ratio=1.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Statistical outlier removal: {len(ind)} points remain")
    return cl

# 3. 下采样（体素网格）
def downsample_point_cloud(pcd, voxel_size=0.01):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel downsampled: {len(pcd_down.points)} points remain")
    return pcd_down

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

    # 步骤1：加载
    pcd = load_point_cloud(args.input)
    print(f"加载后点数: {len(pcd.points)}")
    # 步骤2：去噪
    pcd = denoise_point_cloud(pcd)
    print(f"去噪后点数: {len(pcd.points)}")
    # 保存去噪后的点云
    denoise_out = args.output.rsplit('.', 1)[0] + '.denoise.xyz'
    save_result(pcd, denoise_out)
    # 步骤3：下采样
    pcd = downsample_point_cloud(pcd, voxel_size=args.voxel_size)
    print(f"下采样后点数: {len(pcd.points)}")
    # 保存下采样后的点云
    downsample_out = args.output.rsplit('.', 1)[0] + '.downsample.xyz'
    save_result(pcd, downsample_out)
    # 下采样后估算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    print("下采样后已估算法线")
    # 步骤4：Poisson重建
    mesh = fill_holes_poisson(pcd, depth=args.poisson_depth)
    # 步骤5：从网格表面采样点云
    sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=3000)
    print(f"网格采样后点数: {len(sampled_pcd.points)}")
    # 步骤6：保存结果
    # 保存网格（.ply）
    mesh_out = args.output
    if not mesh_out.endswith('.ply'):
        mesh_out = mesh_out.rsplit('.', 1)[0] + '.ply'
    save_result(mesh, mesh_out)
    # 保存点云（.xyz）
    pcd_out = args.output
    if not pcd_out.endswith('.xyz'):
        pcd_out = pcd_out.rsplit('.', 1)[0] + '.xyz'
    save_result(sampled_pcd, pcd_out)
