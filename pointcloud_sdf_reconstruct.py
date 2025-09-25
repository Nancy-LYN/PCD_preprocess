import pymeshlab
import open3d as o3d
import numpy as np
import argparse
import os
from scipy.spatial import cKDTree
from skimage import measure

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def save_mesh_o3d(mesh, out_path):
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved mesh to {out_path}")

def fill_holes_with_pymeshlab(input_mesh_path, output_mesh_path, max_hole_size=10000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh_path)
    ms.apply_filter('meshing_close_holes', maxholesize=max_hole_size)
    ms.save_current_mesh(output_mesh_path)
    print(f"[pymeshlab] Saved mesh with holes filled to {output_mesh_path}")
    # 用Open3D重新加载，补全法向量并保存ply
    try:
        mesh_filled = o3d.io.read_triangle_mesh(output_mesh_path)
        mesh_filled.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_mesh_path, mesh_filled)
        print(f"[Open3D] Recomputed normals and saved mesh to {output_mesh_path}")
    except Exception as e:
        print(f"[Warning] Open3D normal computation failed: {e}")

def estimate_sdf(points, grid_points, k=20):
    # 用k近邻估算SDF，若有法线则用符号
    tree = cKDTree(points)
    dists, idxs = tree.query(grid_points, k=k)
    sdf = dists.mean(axis=1)
    # 若点云有法线，则用法线方向判定符号
    if hasattr(estimate_sdf, 'normals') and estimate_sdf.normals is not None:
        normals = estimate_sdf.normals
        # 取最近点的法线
        closest_normals = normals[idxs[:,0]]
        # 方向：grid_point - 最近点
        dirs = grid_points - points[idxs[:,0]]
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        sign = np.sign(np.sum(dirs * closest_normals, axis=1))
        sdf = sdf * sign
    return sdf

def sdf_surface_reconstruction(pcd, voxel_size=0.01, padding=5):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0) - voxel_size * padding
    max_bound = points.max(axis=0) + voxel_size * padding
    grid_x, grid_y, grid_z = [np.arange(min_b, max_b, voxel_size) for min_b, max_b in zip(min_bound, max_bound)]
    grid = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), -1)
    grid_points = grid.reshape(-1, 3)
    print(f"SDF grid shape: {grid.shape}, total points: {grid_points.shape[0]}")
    # 若点云有法线，传递给estimate_sdf
    if hasattr(pcd, 'normals') and len(pcd.normals) == len(pcd.points):
        estimate_sdf.normals = np.asarray(pcd.normals)
    else:
        estimate_sdf.normals = None
    sdf = estimate_sdf(points, grid_points, k=20)
    sdf_grid = sdf.reshape(grid.shape[:-1])
    # marching_cubes: level更贴近点云表面
    level = np.percentile(sdf, 30)
    verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=level, spacing=(voxel_size, voxel_size, voxel_size))
    verts = verts + min_bound / voxel_size  # 还原到原始坐标
    verts = verts * voxel_size
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    # 自动补洞（Open3D 0.17+ 支持）
    if hasattr(mesh, 'fill_holes'):
        mesh = mesh.fill_holes()
    # 可选：轻度Laplacian平滑（去锯齿但保留细节）
    try:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
        mesh.compute_vertex_normals()
        print("[Info] 已进行轻度Laplacian平滑")
    except Exception as e:
        print(f"[Warning] Laplacian平滑失败: {e}")

    # 进一步检测大空洞并用法线插值补洞
    try:
        # 1. 找到所有边界环
        boundaries = mesh.get_boundaries()
        for boundary in boundaries:
            if len(boundary) < 10:
                continue  # 小洞跳过
            # 2. 获取边界点坐标
            boundary_points = np.asarray(mesh.vertices)[boundary]
            # 3. 获取边界点法线
            boundary_normals = np.asarray(mesh.vertex_normals)[boundary]
            # 4. 计算边界中心和平均法线
            center = boundary_points.mean(axis=0)
            avg_normal = boundary_normals.mean(axis=0)
            avg_normal /= np.linalg.norm(avg_normal) + 1e-8
            # 5. 在边界内插值生成新点（简单扇形三角剖分）
            new_verts = []
            new_faces = []
            center_idx = len(mesh.vertices)
            mesh.vertices.append(o3d.utility.Vector3dVector([center]))
            for i in range(len(boundary)):
                i_next = (i + 1) % len(boundary)
                new_faces.append([boundary[i], boundary[i_next], center_idx])
            # 6. 合并新面片
            mesh.triangles.extend(o3d.utility.Vector3iVector(new_faces))
            # 7. 给新点赋平均法线
            mesh.vertex_normals.append(o3d.utility.Vector3dVector([avg_normal]))
        mesh.compute_vertex_normals()
    except Exception as e:
        print(f"[Warning] 自动大空洞补洞失败: {e}")
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云SDF重建光滑封闭表面（支持更高精度，推荐voxel_size 0.005~0.01）")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径（.xyz）')
    parser.add_argument('--output', type=str, required=True, help='输出网格文件路径（.ply）')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='SDF体素大小，越小越精细，推荐0.005~0.01')
    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    print(f"Loaded point cloud: {args.input}, 点数: {len(pcd.points)}")
    mesh = sdf_surface_reconstruction(pcd, voxel_size=args.voxel_size)
    print(f"SDF mesh: 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    # 保存补洞前模型
    output_preholes = args.output.replace('.ply', '_preholes.ply')
    save_mesh_o3d(mesh, output_preholes)
    # 用pymeshlab补洞并保存
    fill_holes_with_pymeshlab(output_preholes, args.output, max_hole_size=10000)

# 用法示例：
# python pointcloud_sdf_reconstruct.py --input input/xxx.xyz --output outs/xxx_sdf.ply --voxel_size 0.01
