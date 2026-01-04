import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from model import *
from tools import *
from evaluation_tools import evaluate_PCT, evaluate_CDRL
import gym
from arguments import get_args
import torch
torch.set_printoptions(threshold=np.inf)

def main(args):
    # The name of this evaluation, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\evaluation' and '.\logs\runs'

    if args.custom is not None:
        custom = args.custom
    else:
        custom = input('Please input the evaluate name\n')

    timeStr = custom
    if not args.without_time_str:
        timeStr += '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    args.timeStr = timeStr
    # accelerator = Accelerator()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)
        # device = accelerator.device

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create single packing environment and load existing dataset.
    envs = gym.make(args.id, args = args)


    # Create the main actor & critic networks of PCT
    if args.drl_method != 'rainbow':
        if args.model_architecture == 'PCT':
            PCT_policy =  DRL_GAT(args)
            PCT_policy =  PCT_policy.to(device)
            # PCT_policy = accelerator.prepare(PCT_policy)
            # Load the trained model, if needed
            if args.load_model:
                PCT_policy = load_policy(args.model_path, PCT_policy, args)
                print('Loading pre-train model', args.model_path)
        else:
            import CDRL_model

            if args.setting == 1: channel = 6
            if args.setting == 2: channel = 10
            if args.setting == 3: channel = 7
            if 'category' in args.practical_constrain:
                channel += 1
            PCT_policy = CDRL_model.Policy(
                envs.observation_space.shape, envs.action_space, channel=channel,
                base_kwargs={'recurrent': False, 'hidden_size': 256}).to(device)
            if args.load_model:
                PCT_policy = load_policy(args.model_path, PCT_policy, args)
                print('Loading pre-train model', args.model_path)
    else:
        PCT_policy =  RainbowAgent(args)

    # Backup all py file
    backup(timeStr, args, None)
    
    # Perform all evaluation.
    if args.model_architecture == 'CDRL':
        print('CDRL')
        evaluate_CDRL(PCT_policy, envs, timeStr, args, device,
                eval_freq=args.evaluation_episodes)
    elif args.simple_test:
        print('Simple test')
        evaluate_PCT(PCT_policy, envs, timeStr, args, device,
                eval_freq=args.evaluation_episodes, factor=args.normFactor)


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)
