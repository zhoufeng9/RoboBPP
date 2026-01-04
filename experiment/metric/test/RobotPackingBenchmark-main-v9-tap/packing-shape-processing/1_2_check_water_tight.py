import os
import shutil
import trimesh

# source = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-off'
# target_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-not-watertight'
source = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-off_final'
target_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-off_final_not'

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历 source 文件夹中的所有 .off 文件
for f in os.listdir(source):
    if f.endswith('.off'):
        file_path = os.path.join(source, f)
        mesh = trimesh.load_mesh(file_path)

        if not mesh.is_watertight:
            # 将文件复制到目标文件夹
            target_path = os.path.join(target_dir, f)
            shutil.copy(file_path, target_path)
            print(f'Copied {file_path} to {target_path}')

            # 进行后续处理
            # 这里可以添加你需要的后续处理代码
            # 例如：
            print(f'{f} is not watertight')

print("Done!")
# # 示例的后续处理代码
# for f in os.listdir(target_dir):
#     if f.endswith('.off'):
#         file_path = os.path.join(target_dir, f)
#         mesh = trimesh.load_mesh(file_path)
#         # 这里可以添加你需要的其他处理逻辑
#         print(f'Processed {file_path}, watertight: {mesh.is_watertight}')
