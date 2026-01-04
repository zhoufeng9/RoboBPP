import sys
import torch.cuda
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from model import *
from tools import *
from envs import make_vec_envs
import numpy as np
import random
from train_tools import train_tools
from tensorboardX import SummaryWriter
from tools import registration_envs
from arguments import get_args
from models.model_Attend2Pack import Attend2Pack
from models.model_RCQL import RCQL

# PackE CDRL RCQL Attend2Pack
# --setting 2 --custom packe_pg --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution pg --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/time_series/pg.xlsx
# --setting 2 --custom packe_deli --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution deli --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/occupancy/deli.xlsx
# --setting 2 --custom packe_opai --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution opai --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/flat_long/opai.txt

def main(args):

    if args.custom is not None:
        custom = args.custom
    else:
        custom = input('Please input the experiment name\n')

    # custom = "debug-offline"
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)
    args.device = device

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args, None)
    log_writer_path = './logs/runs/{}'.format('PCT-' + timeStr)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    # Create parallel packing environments to collect training samples online
    envs = make_vec_envs(args, './logs/runinfo', True)

    # Create the main actor & critic networks of PCT
    if args.model_architecture == 'PCT':
        packing_policy =  DRL_GAT(args)
        packing_policy =  packing_policy.to(device)
    elif args.model_architecture == 'CDRL':
        import CDRL_model
        if args.setting == 1: channel = 6
        if args.setting == 2: channel = 6
        if args.setting == 3: channel = 7
        packing_policy = CDRL_model.Policy(
            envs.observation_space.shape, envs.action_space, channel=channel, container_size=args.container_size,
            base_kwargs={'recurrent': False, 'hidden_size' : 256}).to(device)
    elif args.model_architecture == 'PackE':
        import PackE_model
        packing_policy = PackE_model.Policy(
            envs.observation_space.shape, envs.action_space, args.container_size,
            base_kwargs={'recurrent': False, 'hidden_size' : 256}).to(device)

    elif args.model_architecture == 'Attend2Pack':
        originaldim = 3
        embedingdim = 128
        num_head = 8
        FFNdim = 512
        AFFNnum = 3
        inchannel = 2
        outchannel = 4
        C = 10
        outdim = 2 * args.container_size[0]
        L, W = args.container_size[0:2]
        scale = 1 / math.sqrt(embedingdim // num_head)
        packing_policy = Attend2Pack(originaldim, embedingdim, num_head, FFNdim, AFFNnum, inchannel, outchannel, C, outdim,
                                  L, W, scale).to(device)
    elif args.model_architecture == 'RCQL':
        state_size = 6
        hidden_size = 128
        nb_heads = 8
        encoder_nb_layers = 3
        attn_span = 20
        inner_hidden_size = 512
        src_head_hidden_size = 128
        pos_head_hidden_size = 512
        s_res_size = 1
        r_res_size = 2
        x_res_size, y_res_size = args.container_size[0:2]
        decoder_nb_layers = 1
        item_state_size = 3
        packing_policy = RCQL(state_size, hidden_size, nb_heads, encoder_nb_layers, attn_span, inner_hidden_size,
                            src_head_hidden_size, pos_head_hidden_size, s_res_size, r_res_size,
                            x_res_size, y_res_size, decoder_nb_layers, item_state_size).to(device)
    else:
        assert False

    # Load the trained model, if needed
    # args.load_model = True
    if args.load_model:
        # args.model_path = '/media/wzf/goodluck/Workspace/rl/packe/packe_pg_2025.01.15-05-56-56.pt'
        packing_policy = load_policy(args.model_path, packing_policy, args)
        print('Loading pre-train model', args.model_path)


    # Perform all training.
    trainTool = train_tools(writer, timeStr, packing_policy, args)
    if args.model_architecture == 'CDRL':
        trainTool.train_CDRL(envs, args)
    elif args.model_architecture == 'PCT':
        trainTool.train_PCT(envs, args)
    elif args.model_architecture == 'Attend2Pack':
        trainTool.train_Attend2Pack(envs, args)
    elif args.model_architecture == 'RCQL':
        trainTool.train_RCQL(envs, args)
    elif args.model_architecture == 'PackE':
        trainTool.train_PackE(envs, args)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)