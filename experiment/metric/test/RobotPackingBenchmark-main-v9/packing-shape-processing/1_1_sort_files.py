import os
import trimesh


def get_obj_files(root_dir):
    obj_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.obj'):
                obj_files.append(os.path.join(dirpath, filename))
    return obj_files


def write_off(filename, mesh):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(mesh.vertices)} {len(mesh.faces)} 0\n")
        for vertex in mesh.vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def convert_obj_to_off(obj_files, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for obj_file in obj_files:
        try:
            # Load the mesh or scene from the .obj file
            loaded = trimesh.load(obj_file)

            # Get the parent directory name
            parent_dir_name = os.path.basename(os.path.dirname(obj_file))

            # Construct the output .off file path
            off_file = os.path.join(target_dir, f"{parent_dir_name}.off")

            # Check if the loaded object is a Scene
            if isinstance(loaded, trimesh.Scene):
                # Merge all geometries in the scene into a single mesh
                combined_mesh = trimesh.util.concatenate(loaded.dump())
                # Export the combined mesh to the .off file using the custom write_off function
                write_off(off_file, combined_mesh)
            else:
                # If it's a single Trimesh object
                write_off(off_file, loaded)

            print(f'Successfully converted {obj_file} to {off_file}')
        except Exception as e:
            print(f'Failed to convert {obj_file}: {e}')


# 指定文件夹路径
root_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/data-mesh'

# 获取所有.obj文件的路径
obj_files = get_obj_files(root_dir)

# 指定目标文件夹路径
target_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-off'

# 转换所有.obj文件为.off文件
convert_obj_to_off(obj_files, target_dir)
print("Done!")
