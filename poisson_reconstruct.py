import open3d as o3d
import argparse
import os

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def save_mesh(mesh, out_path):
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved Poisson reconstructed mesh to {out_path}")

def poisson_reconstruction(pcd, depth=9):
    # 法线估算（强制）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    # 法线方向一致性将在主程序中调用，避免状态异常
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.compute_vertex_normals()
    return mesh, densities

def crop_mesh_by_density(mesh, densities, keep_ratio=0.95):
    # 去除密度最低的部分（通常是伪影）
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 1 - keep_ratio)
    vertices_to_keep = densities > threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="泊松重建封闭网格")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径（.xyz/.ply）')
    parser.add_argument('--output', type=str, required=True, help='输出网格文件路径（.ply）')
    parser.add_argument('--depth', type=int, default=9, help='泊松重建深度（越大越细致，建议8~12）')
    parser.add_argument('--keep_ratio', type=float, default=0.95, help='保留高密度顶点比例（去除伪影）')
    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    print(f"Loaded point cloud: {args.input}, 点数: {len(pcd.points)}")
    # 法线估算和方向一致性
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    # 强制写入法线属性
    import numpy as np
    if not pcd.has_normals() or len(np.asarray(pcd.normals)) != len(pcd.points):
        print("[Error] 点云法线估算失败，无法进行泊松重建。请检查输入点云。")
        exit(1)
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except Exception as e:
        print(f"[Warning] 法线方向一致性处理失败: {e}")
    if not pcd.has_normals() or len(np.asarray(pcd.normals)) != len(pcd.points):
        print("[Error] 点云法线属性缺失，无法进行泊松重建。请检查输入点云。")
        exit(1)
    mesh, densities = poisson_reconstruction(pcd, depth=args.depth)
    print(f"Poisson mesh: 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    if args.keep_ratio < 1.0:
        mesh = crop_mesh_by_density(mesh, densities, keep_ratio=args.keep_ratio)
        print(f"Cropped mesh: 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    save_mesh(mesh, args.output)

# 用法示例：
# python poisson_reconstruct.py --input input/xxx.xyz --output outs/xxx_poisson.ply --depth 10 --keep_ratio 0.95
