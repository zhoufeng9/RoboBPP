import os
import trimesh

# Remove the meshes that are not watertight after reconstruction.
source = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-off-scaled_final_good'

for f in os.listdir(source):
    print(f)
    mesh = trimesh.load_mesh(os.path.join(source,f))
    if not mesh.is_watertight:
        os.remove(os.path.join(source, f))

