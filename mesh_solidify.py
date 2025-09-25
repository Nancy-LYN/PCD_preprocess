import open3d as o3d
import numpy as np
import argparse
import os
from scipy.ndimage import binary_fill_holes

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False
    print("[Warning] pymeshlab not installed, solidify method 3 will be skipped.")

def load_mesh(file_path):
    return o3d.io.read_triangle_mesh(file_path)

def save_mesh(mesh, out_path):
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved mesh to {out_path}")

def save_point_cloud(pcd, out_path):
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved point cloud to {out_path}")

def voxel_fill_solid(mesh, voxel_size=0.01):
    # 1. 网格体素化
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    if len(voxels) == 0:
        raise RuntimeError("Voxelization failed: no voxels found.")
    min_idx = voxels.min(axis=0)
    max_idx = voxels.max(axis=0)
    grid_shape = max_idx - min_idx + 1
    grid = np.zeros(grid_shape, dtype=bool)
    grid_idx = voxels - min_idx
    grid[grid_idx[:,0], grid_idx[:,1], grid_idx[:,2]] = True
    # 2. 填充空心体
    grid_filled = binary_fill_holes(grid)
    filled_indices = np.argwhere(grid_filled)
    filled_points = filled_indices + min_idx
    filled_points = filled_points * voxel_size + voxel_size / 2 + np.asarray(voxel_grid.origin)
    # 3. 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filled_points)
    # 4. 网格（Marching Cubes）
    try:
        import skimage.measure
        verts, faces, _, _ = skimage.measure.marching_cubes(grid_filled.astype(float), 0.5)
        verts = verts + min_idx
        verts = verts * voxel_size + voxel_size / 2 + np.asarray(voxel_grid.origin)
        mesh_filled = o3d.geometry.TriangleMesh()
        mesh_filled.vertices = o3d.utility.Vector3dVector(verts)
        mesh_filled.triangles = o3d.utility.Vector3iVector(faces)
        # 优化：自动修正法线方向
        mesh_filled.orient_triangles()
        mesh_filled.compute_triangle_normals()
        mesh_filled.compute_vertex_normals()
    except ImportError:
        print("[Warning] skimage not installed, mesh output skipped for method 1.")
        mesh_filled = None
    return pcd, mesh_filled

def solidify_mesh_pymeshlab(mesh, voxel_size=0.01):
    if not HAS_PYMESHLAB:
        print("[Warning] pymeshlab not available, skipping solidify.")
        return None, None
    # 保存临时文件
    tmp_in = "_tmp_in.ply"
    tmp_out = "_tmp_out.ply"
    o3d.io.write_triangle_mesh(tmp_in, mesh)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tmp_in)
    # 多次close_holes_simple
    for _ in range(3):
        ms.apply_filter('close_holes_simple')
    # solidify，offset可调
    ms.apply_filter('solidify', solidifytype=0, offset=0.02, multi=False)
    # 自动修正法线方向
    ms.apply_filter('re_orient_all_faces_coherently')
    ms.save_current_mesh(tmp_out)
    mesh_solid = o3d.io.read_triangle_mesh(tmp_out)
    mesh_solid.orient_triangles()
    mesh_solid.compute_triangle_normals()
    mesh_solid.compute_vertex_normals()
    # 采样点云
    pcd = mesh_solid.sample_points_poisson_disk(number_of_points=50000)
    # 清理临时文件
    os.remove(tmp_in)
    os.remove(tmp_out)
    return pcd, mesh_solid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="空心网格实心化（体素填充/solidify）并输出点云和网格")
    parser.add_argument('--input', type=str, required=True, help='输入网格文件路径（.ply）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='体素大小')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mesh = load_mesh(args.input)
    print(f"Loaded mesh: {args.input}, 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")

    # 方法1：体素填充
    print("[Method 1] 体素填充实心体...")
    pcd1, mesh1 = voxel_fill_solid(mesh, voxel_size=args.voxel_size)
    save_point_cloud(pcd1, os.path.join(args.output_dir, 'solid_voxel_points.xyz'))
    if mesh1 is not None:
        save_mesh(mesh1, os.path.join(args.output_dir, 'solid_voxel_mesh.ply'))

    # 方法3：pymeshlab solidify
    if HAS_PYMESHLAB:
        print("[Method 3] pymeshlab solidify...")
        pcd3, mesh3 = solidify_mesh_pymeshlab(mesh, voxel_size=args.voxel_size)
        if pcd3 is not None:
            save_point_cloud(pcd3, os.path.join(args.output_dir, 'solid_pymeshlab_points.xyz'))
        if mesh3 is not None:
            save_mesh(mesh3, os.path.join(args.output_dir, 'solid_pymeshlab_mesh.ply'))
    else:
        print("[Method 3] pymeshlab not available, skipped.")

# 用法示例：
# python mesh_solidify.py --input input/xxx.ply --output_dir outs/xxx_solid --voxel_size 0.01
