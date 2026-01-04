import os
import torch
import numpy as np
import trimesh

bin_dimension = [0.32,0.32,0.3]
bin_scale = np.linalg.norm(bin_dimension)

# subName = 'ycb'
# baseDir = './final_data/{}/vhacd_with_pose'.format(subName)
baseDir = '/media/wzf/goodluck/Workspace/rl/ARMBench/processing/data-mesh-vhacd_with_pose_all'

id2shape = {}
counter = 0
for item in os.listdir(baseDir):
    print(item)
    mesh = trimesh.load_mesh(os.path.join(baseDir,item))
    id2shape[counter] = item
    counter += 1

if not os.path.exists('/media/wzf/goodluck/Workspace/rl/ARMBench/armbench/id2shape.pt'):
    torch.save(id2shape,'/media/wzf/goodluck/Workspace/rl/ARMBench/armbench/id2shape.pt')
else:
    id2shape = torch.load('/media/wzf/goodluck/Workspace/rl/ARMBench/armbench/id2shape.pt')

inverseDict = {}
for k in id2shape.keys():
    if id2shape[k][0:-6] not in inverseDict.keys():
        inverseDict[id2shape[k][0:-6]] = [k]
    else:
        inverseDict[id2shape[k][0:-6]].append(k)
print(inverseDict)
print(len(inverseDict.keys()))

allTrajs = []
for i in range(10000):
    print(i)
    traj = []
    for j in range(100):
        name = np.random.choice(list(inverseDict.keys()))
        traj.append(np.random.choice(inverseDict[name]))
    allTrajs.append(traj.copy())
torch.save(allTrajs, '/media/wzf/goodluck/Workspace/rl/ARMBench/armbench/test_sequence.pt')