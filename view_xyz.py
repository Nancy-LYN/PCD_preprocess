import open3d as o3d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化xyz点云文件")
    parser.add_argument('--input', type=str, required=True, help='输入点云文件路径')
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded point cloud: {args.input}, 点数: {len(pcd.points)}")
    o3d.visualization.draw_geometries([pcd])

# python view_xyz.py --input outs/test6_smooth.xyz