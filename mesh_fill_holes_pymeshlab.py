import pymeshlab
import argparse
import os

def fill_holes_with_pymeshlab(input_mesh, output_mesh, max_hole_size=10000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh)
    # 自动补洞，max_hole_size越大可补更大空洞
    ms.apply_filter('meshing_close_holes', maxholesize=max_hole_size)
    ms.save_current_mesh(output_mesh)
    print(f"Saved mesh with holes filled to {output_mesh}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用pymeshlab自动补洞（大空洞）")
    parser.add_argument('--input', type=str, required=True, help='输入网格文件路径（.ply）')
    parser.add_argument('--output', type=str, required=True, help='输出补洞后网格文件路径（.ply）')
    parser.add_argument('--max_hole_size', type=int, default=10000, help='最大补洞面片大小')
    args = parser.parse_args()

    fill_holes_with_pymeshlab(args.input, args.output, max_hole_size=args.max_hole_size)

# 用法示例：
# python mesh_fill_holes_pymeshlab.py --input input/xxx.ply --output outs/xxx_filled.ply --max_hole_size 10000
