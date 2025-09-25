import open3d as o3d
import argparse
import os

def load_mesh(file_path):
    return o3d.io.read_triangle_mesh(file_path)

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved surface point cloud to {out_path}")

def sample_surface_points(mesh, num_points=3000):
    # 只在mesh表面采样，得到单层点云
    return mesh.sample_points_poisson_disk(number_of_points=num_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对mesh模型采样单层表面点云")
    parser.add_argument('--input', type=str, required=True, help='输入网格文件路径（.ply）')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件路径（.xyz）')
    parser.add_argument('--num_points', type=int, default=3000, help='采样点数')
    args = parser.parse_args()

    mesh = load_mesh(args.input)
    print(f"Loaded mesh: {args.input}, 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    pcd = sample_surface_points(mesh, num_points=args.num_points)
    print(f"Sampled surface point cloud: {len(pcd.points)} points")
    save_point_cloud(pcd, args.output)

# 用法示例：
# python mesh_surface_sample.py --input outs/test6_alphaShape/test6_alphaShape_alpha_shape_filled.ply --output outs/test6_alphaShape/surface_sampled.xyz --num_points 3000
