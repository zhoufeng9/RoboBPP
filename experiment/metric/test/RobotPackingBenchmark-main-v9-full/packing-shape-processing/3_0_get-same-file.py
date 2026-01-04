import os

# 定义文件夹路径
final_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-final'
vhacd_dir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-vhacd'

# 获取data-mesh-final中的所有文件名（不包括扩展名）
final_files = {os.path.splitext(f)[0] for f in os.listdir(final_dir) if os.path.isfile(os.path.join(final_dir, f))}

# 遍历data-mesh-vhacd中的所有文件
for f in os.listdir(vhacd_dir):
    file_path = os.path.join(vhacd_dir, f)
    file_name_without_ext = os.path.splitext(f)[0]

    # 检查文件是否在final_files中
    if file_name_without_ext not in final_files:
        # 如果文件不在final_files中，则删除该文件
        os.remove(file_path)
        print(f'Deleted {file_path}')
