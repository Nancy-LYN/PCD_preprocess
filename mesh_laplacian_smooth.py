import open3d as o3d
import argparse
import os

def load_mesh(file_path):
    return o3d.io.read_triangle_mesh(file_path)

def save_mesh(mesh, out_path):
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved smoothed mesh to {out_path}")

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved resampled point cloud to {out_path}")

def laplacian_smooth(mesh, iterations=20, lambda_coef=0.5):
    # Open3D Laplacian smooth: preserves overall shape, smooths local noise
    mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=iterations, lambda_filter=lambda_coef)
    mesh_out.compute_vertex_normals()
    # 去除异常面片
    mesh_out.remove_degenerate_triangles()
    mesh_out.remove_duplicated_triangles()
    mesh_out.remove_non_manifold_edges()
    mesh_out.remove_duplicated_vertices()
    # 填补空洞（Open3D 0.17+ 支持）
    if hasattr(mesh_out, 'fill_holes'):
        mesh_out = mesh_out.fill_holes()
    return mesh_out

def sample_mesh_points(mesh, num_points=3000):
    return mesh.sample_points_poisson_disk(number_of_points=num_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网格Laplacian平滑+点云重采样")
    parser.add_argument('--input', type=str, required=True, help='输入网格文件路径（.ply）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹')
    parser.add_argument('--iterations', type=int, default=20, help='Laplacian平滑迭代次数')
    parser.add_argument('--lambda_coef', type=float, default=0.5, help='Laplacian平滑系数')
    parser.add_argument('--num_points', type=int, default=3000, help='重采样点云数量')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mesh = load_mesh(args.input)
    print(f"Loaded mesh: {args.input}, 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")

    # Laplacian平滑
    mesh_smooth = laplacian_smooth(mesh, iterations=args.iterations, lambda_coef=args.lambda_coef)
    mesh_smooth_path = os.path.join(args.output_dir, 'smoothed_mesh.ply')
    save_mesh(mesh_smooth, mesh_smooth_path)

    # 点云重采样
    pcd = sample_mesh_points(mesh_smooth, num_points=args.num_points)
    pcd_path = os.path.join(args.output_dir, 'smoothed_resampled.xyz')
    save_point_cloud(pcd, pcd_path)

# 用法示例：
# python mesh_laplacian_smooth.py --input outs/test6_alphaShape/test6_alphaShape_alpha_shape_filled.ply --output_dir outs/test6_alphaShape_LL --iterations 20 --lambda_coef 0.5 --num_points 3000
