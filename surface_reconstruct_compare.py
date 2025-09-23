import open3d as o3d
import numpy as np
import argparse
import os

def load_point_cloud(file_path):
    base_dir = os.path.dirname(__file__)
    if file_path.startswith("/input/"):
        file_path = os.path.join(base_dir, "input", os.path.basename(file_path))
    elif file_path.startswith("/outs/"):
        file_path = os.path.join(base_dir, "outs", os.path.basename(file_path))
    return o3d.io.read_point_cloud(file_path)

def save_result(obj, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    if isinstance(obj, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(out_path, obj)
        print(f"Saved point cloud to {out_path}")
    elif isinstance(obj, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(out_path, obj)
        print(f"Saved mesh to {out_path}")
    else:
        print("Unknown object type for saving.")

def alpha_shape_mesh(pcd, alpha=0.5):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    print(f"Alpha Shape mesh created with alpha={alpha}")
    return mesh

def ball_pivoting_mesh(pcd, radii=[0.005, 0.01, 0.02]):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    print(f"Ball Pivoting mesh created with radii={radii}")
    return mesh

def sample_mesh_points(mesh, num_points=3000):
    return mesh.sample_points_poisson_disk(number_of_points=num_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云Alpha Shapes与Ball Pivoting重建")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha Shapes参数')
    parser.add_argument('--bp_radii', type=float, nargs='+', default=[0.005,0.01,0.02], help='Ball Pivoting半径列表')
    args = parser.parse_args()

    # 加载点云
    pcd = load_point_cloud(args.input)
    print(f"Loaded point cloud: {args.input}, 点数: {len(pcd.points)}")

    # 输出文件夹
    out_base = os.path.dirname(args.output)
    out_name = os.path.splitext(os.path.basename(args.output))[0]
    out_dir = os.path.join(out_base, out_name)
    os.makedirs(out_dir, exist_ok=True)

    # Alpha Shapes
    mesh_alpha = alpha_shape_mesh(pcd, alpha=args.alpha)
    save_result(mesh_alpha, out_dir, f'{out_name}_alpha_shape.ply')
    pcd_alpha = sample_mesh_points(mesh_alpha, num_points=3000)
    print(f"Alpha Shape采样点云数量: {len(pcd_alpha.points)}")
    save_result(pcd_alpha, out_dir, f'{out_name}_alpha_shape.xyz')

    # Ball Pivoting
    mesh_bp = ball_pivoting_mesh(pcd, radii=args.bp_radii)
    save_result(mesh_bp, out_dir, f'{out_name}_ball_pivoting.ply')
    pcd_bp = sample_mesh_points(mesh_bp, num_points=3000)
    print(f"Ball Pivoting采样点云数量: {len(pcd_bp.points)}")
    save_result(pcd_bp, out_dir, f'{out_name}_ball_pivoting.xyz')

# python surface_reconstruct_compare.py --input input/xxx.xyz --output outs/xxx_compare --alpha 0.5 --bp_radii 0.005 0.01 0.02