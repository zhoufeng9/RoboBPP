import os
import torch
from shutil import copyfile, copytree
import torch.nn as nn
import numpy as np
from gym.envs.registration import register

# from   vis_boxes import vis_internal_nodes
# import vispy.scene
# import vispy.io

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 1:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 3:
            bias = self._bias.t().view(1, 1, -1)
        elif x.dim() == 4:
            bias = self._bias.t().view(1, -1, 1, 1)
        else:
            assert False

        return x + bias

def backup(timeStr, args, upper_policy = None):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists('./logs/runinfo'):
        os.makedirs('./logs/runinfo')

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    for file in os.listdir('./'):
        if file.endswith('.py'):
            copyfile(file, os.path.join(targetDir, file))

    gymPath = './pct_envs'
    envName = args.id.split('-v')
    envName = envName[0] + envName[1]
    envPath = os.path.join(gymPath, envName)
    try:
        copytree(envPath, os.path.join(targetDir, envName))
    except:
        pass

    # save args to args.txt
    with open(os.path.join(targetDir, "args.txt"), "w") as f_args:
        for k, v in args.__dict__.items():
            f_args.write("%s:%s\n"%(k, v))

    if upper_policy is not None:
        torch.save(upper_policy.state_dict(), os.path.join(args.model_save_path, timeStr, 'upper-first-' + timeStr + ".pt"))
    # export args to args.txt
    with open(os.path.join(targetDir, "args.txt"), "w") as f_args:
        for k, v in args.__dict__.items():
            f_args.write("%s:%s\n"%(k, v))

# Parsing PCT node from state returned in environment
def get_leaf_nodes(observation, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((observation.shape[0], -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes

def get_leaf_nodes_with_factor(observation,  batch_size, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((batch_size, -1, 9))
    # unify_obs[:, :, 0:6] *= factor
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes

'''
Parsing the raw state returned in environment:

internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                    x1, y1, z1 are coordinates of a packed item
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are sizes of a packed item (a little different from the original paper,
                    these two description have similar performance.).
leaf_nodes        : A placement vector, [x1, y1, z1, x2, y2, z2]
                    x1, y1, z1 are coordinates of a placement.
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are  sizes of the current item after an axis-aligned orientation (a little different from the original paper,
                    these two description have similar performance.).
next_item         : The next item to be packed [density(optional), 0, 0,x, y, z]
                    x, y, z are  sizes of the current item.
invalid_leaf_nodes: The mask which indicates whether this placement is feasible.
full_mask         : The mask which indicates whether this node should be encode by GAT.
'''
def observation_decode_leaf_node(observation, internal_node_holder, internal_node_length, leaf_node_holder):
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes  = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    current_box = observation[:,internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag  = observation[:,internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    full_mask   = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask

def recursive_plot_ems(cur_node, ax, x, y, dx, dy, rec_cnt):
    if rec_cnt > 6:
        return
    if cur_node.sub_container:
        is_subc = "True"
    else:
        is_subc = "False"
    info = "nodes: " + str(cur_node.tree_node_num) + "\n" +\
            "children: " + str(cur_node.valid_children) + "\n" +\
            "objs: " + str(len(cur_node.placed_items)) + "\n"+\
            "is_subc: " + is_subc + "\n"+\
            "lnodes" + str(len(cur_node.leaf_nodes)) + "\n"
    cnt = 0
    for item in cur_node.placed_items:
        tmp = list(item)
        for i in range(len(item)):
            tmp[i] = round(item[i], 3)
        if cnt % 2:
            info = info + str(tmp) + "\n"
        else:
            info = info + str(tmp) + " "
        cnt += 1

    ax.text(x, y, info, bbox=dict(facecolor='white', alpha=1), ha='center', va='center')

    if cur_node.children:
        next_dx = dx / len(cur_node.children)
        next_x = x - dx / 2 + next_dx / 2

        next_y = y - dy
        for child in cur_node.children:
            ax.plot([x, next_x], [y, next_y], 'k-')
            recursive_plot_ems(child, ax, next_x, next_y, next_dx, dy, rec_cnt+1)
            next_x += next_dx   

def load_policy(load_path, upper_policy, args):
    print(load_path)
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = torch.load(load_path, map_location='cpu')
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
                load_dict[k.replace('module.', '')] = v
    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}

    if args.model_architecture == 'PCT':
        load_dict = {k.replace('critic', 'critic_moduel'): v for k, v in load_dict.items()}
        load_dict = {k.replace('critic_moduel_moduel', 'critic_moduel'): v for k, v in load_dict.items()}
        load_dict = {k.replace('critic_moduel_distill', 'critic_distill'): v for k, v in load_dict.items()}

    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)

    upper_policy.load_state_dict(load_dict, strict=True)

    print('Loading pre-train upper model', load_path)

    return upper_policy


def draw(internal_nodes, img_idx, img_path, root_path, container_size=[1, 1, 1], container_position=[0, 0, 0], bin_size=[1, 1, 1]):
    if not os.path.exists(os.path.join(root_path, str(img_idx))):
        os.makedirs(os.path.join(root_path, str(img_idx)))
    img_path = os.path.join(root_path, str(img_idx), img_path)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, size=(1920, 1080))
    view = canvas.central_widget.add_view()
    camera = vispy.scene.cameras.TurntableCamera(fov=60,
                                                 center=container_position,
                                                 distance=2 * np.linalg.norm(container_size))
    view.camera = camera
    vis_internal_nodes(internal_nodes, view, container_size=container_size, container_position=container_position, bin_size=bin_size)
    img = canvas.render()
    vispy.io.write_png(img_path, img)
    canvas.close()

def registration_envs():
    register(
        id='PctDiscrete-v2',                                  # Format should be xxx-v0, xxx-v1
        entry_point='pct_envs.PctDiscrete2:PackingDiscrete',  # Expalined in envs/__init__.py
    )
    register(
        id='PctDiscrete-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='pct_envs.PctDiscrete0:PackingDiscrete',  # Expalined in envs/__init__.py
    )
    register(
        id='PctDiscrete-v3',                                  # Format should be xxx-v0, xxx-v1
        entry_point='pct_envs.PctDiscrete3:PackingDiscrete',  # Expalined in envs/__init__.py
    )
    register(
        id='PctContinuous-v0',
        entry_point='pct_envs.PctContinuous0:PackingContinuous',
    )
    register(
        id='PctContinuous-v2',
        entry_point='pct_envs.PctContinuous2:PackingContinuous'
    )