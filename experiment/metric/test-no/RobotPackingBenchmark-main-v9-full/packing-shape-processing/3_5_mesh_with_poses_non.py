import os
import shutil

# 定义文件夹路径
vhacd_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-vhacd'
min_area_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-min_area'
vhacd_non_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-vhacd-non'

# 确保目标文件夹存在
if not os.path.exists(vhacd_non_dir):
    os.makedirs(vhacd_non_dir)

# 获取data-mesh-min_area中的所有文件名（不包括扩展名）
min_area_files = {os.path.splitext(f)[0] for f in os.listdir(min_area_dir) if
                  os.path.isfile(os.path.join(min_area_dir, f))}

# 遍历data-mesh-vhacd中的所有.obj文件
for f in os.listdir(vhacd_dir):
    if f.endswith('.obj'):
        file_name_without_ext = os.path.splitext(f)[0]

        # 检查文件名是否在min_area_files中
        if file_name_without_ext not in min_area_files:
            source_path = os.path.join(vhacd_dir, f)
            target_path = os.path.join(vhacd_non_dir, f"{file_name_without_ext}_0.obj")

            # 将文件复制到目标文件夹并重命名
            shutil.copy(source_path, target_path)
            print(f'Copied {source_path} to {target_path}')
