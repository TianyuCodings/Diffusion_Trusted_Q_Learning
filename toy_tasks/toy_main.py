import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append(("../"))

from data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import torch.nn.functional as F
import random

"""If you are using DQL-KL, you can specific generation_sigma when init agent"""
#from agents.dql_kl import DQL_KL as Agent

from agents.dtql import DTQL as Agent

def train_toy_task(args,file_name):
    data_manager = DataGenerator(args.env_name,args.reward_type,args.device)
    data_manager.generate_samples(10000)
    device = args.device

    """Agent Training"""
    agent = Agent(state_dim=data_manager.state_dim,
                  action_dim=data_manager.action_dim,
                  action_space=data_manager.action_space,
                  device=device,
                  discount=args.discount,
                  lr=args.lr,
                  alpha=args.alpha,
                  lr_decay=args.lr_decay,
                  lr_maxt=args.num_epochs * args.num_steps_per_epoch,
                  expectile=args.expectile,
                  sigma_data=args.sigma_data,
                  sigma_max=args.sigma_max,
                  sigma_min=args.sigma_min,
                  tau=args.tau,
                  gamma=args.gamma,
                  repeats=args.repeats)
    if args.pretrain_epochs is not None:
        agent.load_or_pretrain_models(
            dir=file_name,
            replay_buffer=data_manager,
            batch_size=args.batch_size,
            pretrain_steps=args.pretrain_epochs*args.num_steps_per_epoch,
            num_steps_per_epoch=args.num_steps_per_epoch)

    max_iter = args.num_epochs * args.num_steps_per_epoch
    pbar = tqdm(range(max_iter))
    for i in range(max_iter):
        loss_metric = agent.train(replay_buffer=data_manager,
                                  batch_size=args.batch_size)
        pbar.update(1)

    with torch.no_grad():
        state = data_manager.state.reshape(-1, 1).to(agent.device).to(torch.float32)
        try: #for gaussian policy
            new_action = agent.distill_actor.sample(state)
        except: #for implicit policy
            noise = torch.randn((state.shape[0], data_manager.action_dim)) * args.generation_sigma
            noise = noise.to(agent.device)
            new_action = agent.distill_actor(noise, state, torch.tensor([args.generation_sigma]).to(agent.device))
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data_manager.action.cpu().numpy()[:, 0], data_manager.action.cpu().numpy()[:, 1], c=data_manager.reward.cpu().numpy())
        plt.colorbar(scatter, label='Reward values')
        #plt.scatter(action.cpu().numpy()[:, 0], action.cpu().numpy()[:, 1], c="blue",alpha=0.1)
        plt.title(args.env_name)
        plt.xlabel('Action Dimension 1')
        plt.ylabel('Action Dimension 2')
        plt.grid(True)

        new_action = new_action.cpu().numpy()
        plt.scatter(new_action[:, 0], new_action[:, 1], c='red',alpha=0.5)
        # Save the plot to the specified directory
        plot_path = file_name + "/toy.png"
        plt.savefig(plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument("--env_name", default="25_gaussian", type=str, help='25_gaussian or swiss_roll_2D')
    parser.add_argument("--seed", default=1, type=int, help='random seed (default: 0)')
    parser.add_argument("--dir", default="results_toy", type=str)
    parser.add_argument("--reward_type", default="far", type=str, help="can be far, near, and hard.")

    parser.add_argument("--pretrain_epochs", default=5, type=int)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--sigma_max", default=80, type=int)
    parser.add_argument("--sigma_min", default=0.002, type=int)
    parser.add_argument("--sigma_data", default=0.5, type=int)
    parser.add_argument("--generation_sigma", default=2.5, type=float)
    parser.add_argument('--save_checkpoints', action='store_true')

    parser.add_argument("--num_steps_per_epoch", default=100, type=int)
    parser.add_argument("--discount", default=0.99, type=float, help='discount factor for reward (default: 0.99)')

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.lr_decay = False
    args.batch_size = 4096
    args.gamma = 0
    args.expectile = 0.95

    file_name = f'|expect-{args.expectile}'
    file_name += f"|alpha-{args.alpha}|gamma-{args.gamma}"
    file_name += f'|seed={args.seed}'
    file_name += f'|lr={args.lr}'
    if args.lr_decay:
        file_name += f'|lr_decay'
    if args.pretrain_epochs is not None:
        file_name += f'|pretrain={args.pretrain_epochs}'
    file_name += f'|{args.reward_type}'
    #file_name += f'|{args.env_name}'

    results_dir = os.path.join(args.output_dir, args.env_name, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    variant = vars(args)
    variant.update(version=f"DTQL")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_toy_task(args,results_dir)