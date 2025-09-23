import open3d as o3d
import numpy as np
import argparse
import os
import trimesh

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

def postprocess_mesh_remove_sharp(mesh, min_area=1e-4, max_aspect_ratio=20):
    # 分离连通片段
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangles = np.asarray(mesh.triangles)
    to_remove = []
    for i, area in enumerate(cluster_area):
        # 面积太小的片段直接删除
        if area < min_area:
            to_remove.extend(np.where(triangle_clusters == i)[0])
    # 计算每个片段的长宽比，删除极端细长的片段
    # 这里只做面积过滤，长宽比可根据实际需求补充
    if to_remove:
        mesh.remove_triangles_by_index(to_remove)
        mesh.remove_unreferenced_vertices()
        print(f"Removed {len(to_remove)} triangles (small/sharp regions)")
    return mesh

def fill_mesh_holes_with_trimesh(mesh, out_path=None):
    # Open3D mesh -> trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    n_before = tmesh.faces.shape[0]
    n_holes = tmesh.fill_holes()
    n_after = tmesh.faces.shape[0]
    print(f"Filled {n_holes} holes, faces: {n_before} -> {n_after}")
    if out_path:
        tmesh.export(out_path)
        print(f"Saved mesh with holes filled to {out_path}")
    # 转回Open3D mesh
    mesh_filled = o3d.geometry.TriangleMesh()
    mesh_filled.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
    mesh_filled.triangles = o3d.utility.Vector3iVector(tmesh.faces)
    return mesh_filled

def sample_mesh_points(mesh, num_points=3000):
    return mesh.sample_points_poisson_disk(number_of_points=num_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云Alpha Shapes重建+后处理")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha Shapes参数')
    parser.add_argument('--min_area', type=float, default=1e-4, help='后处理最小片段面积阈值')
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
    save_result(mesh_alpha, out_dir, f'{out_name}_alpha_shape_raw.ply')

    # 后处理：删除小面积/尖锐片段
    mesh_alpha_post = postprocess_mesh_remove_sharp(mesh_alpha, min_area=args.min_area)
    save_result(mesh_alpha_post, out_dir, f'{out_name}_alpha_shape_post.ply')

    # 空洞填补
    mesh_alpha_filled = fill_mesh_holes_with_trimesh(mesh_alpha_post, out_path=os.path.join(out_dir, f'{out_name}_alpha_shape_filled.ply'))

    # 采样点云
    pcd_alpha = sample_mesh_points(mesh_alpha_filled, num_points=3000)
    print(f"Alpha Shape空洞填补采样点云数量: {len(pcd_alpha.points)}")
    save_result(pcd_alpha, out_dir, f'{out_name}_alpha_shape_filled.xyz')

# 用法示例：
# python surface_reconstruct_alpha_post.py --input input/xxx.xyz --output outs/xxx_compare --alpha 0.05 --min_area 1e-4
