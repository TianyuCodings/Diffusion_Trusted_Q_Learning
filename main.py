import argparse
import gym
import numpy as np
import os
import torch
from pathlib import Path

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
#from agents.dtql import DTQL as Agent
from agents.dql_kl import DQL_KL as Agent
import random

offline_hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'alpha': 1.0,     'gamma': 0.0, 'lr_decay': False, 'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': False, 'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'alpha': 50.0,    'gamma': 0.0, 'lr_decay': False, 'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'hopper-medium-v2':              {'lr': 1e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': True,  'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': False, 'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'alpha': 20.0,    'gamma': 0.0, 'lr_decay': False, 'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'walker2d-medium-v2':            {'lr': 3e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': True,  'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': True,  'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'alpha': 5.0,     'gamma': 0.0, 'lr_decay': True,  'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 500,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-5, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': True,  'num_epochs': 500,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-medium-play-v0':        {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 400,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 400,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 350,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'alpha': 0.5,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 300,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-umaze-v2':              {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 500,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-umaze-diverse-v2':      {'lr': 3e-5, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': True,  'num_epochs': 500,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-medium-play-v2':        {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 400,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-medium-diverse-v2':     {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 400,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-large-play-v2':         {'lr': 3e-4, 'alpha': 1.0,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 350,  'batch_size': 2048,  'expectile': 0.9},
    'antmaze-large-diverse-v2':      {'lr': 3e-4, 'alpha': 0.5,     'gamma': 1.0, 'lr_decay': False, 'num_epochs': 300,  'batch_size': 2048,  'expectile': 0.9},
    'pen-human-v1':                  {'lr': 3e-5, 'alpha': 1500.0,  'gamma': 0.0, 'lr_decay': True,  'num_epochs': 300,  'batch_size': 256,   'expectile': 0.9},
    'pen-cloned-v1':                 {'lr': 1e-5, 'alpha': 1500.0,  'gamma': 0.0, 'lr_decay': False, 'num_epochs': 200,  'batch_size': 256,   'expectile': 0.7},
    'kitchen-complete-v0':           {'lr': 1e-4, 'alpha': 200.0,   'gamma': 0.0, 'lr_decay': True,  'num_epochs': 500,  'batch_size': 256,   'expectile': 0.7},
    'kitchen-partial-v0':            {'lr': 1e-4, 'alpha': 100.0,   'gamma': 0.0, 'lr_decay': True,  'num_epochs': 1000, 'batch_size': 256,   'expectile': 0.7},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'alpha': 200.0,   'gamma': 0.0, 'lr_decay': True,  'num_epochs': 500,  'batch_size': 256,   'expectile': 0.7},
}

def train_agent(env, state_dim, action_dim, device, output_dir, args):
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  action_space=env.action_space,
                  device=device,
                  discount=args.discount,
                  lr=args.lr,
                  alpha=args.alpha,
                  lr_decay=args.lr_decay,
                  lr_maxt=args.num_epochs*args.num_steps_per_epoch,
                  expectile=args.expectile,
                  sigma_data=args.sigma_data,
                  sigma_max=args.sigma_max,
                  sigma_min=args.sigma_min,
                  tau=args.tau,
                  gamma=args.gamma,
                  repeats=args.repeats)
    if args.pretrain_epochs is not None:
        agent.load_or_pretrain_models(
            dir=str(Path(output_dir)),
            replay_buffer=data_sampler,
            batch_size=args.batch_size,
            pretrain_steps=args.pretrain_epochs*args.num_steps_per_epoch,
            num_steps_per_epoch=args.num_steps_per_epoch)

    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    log_interval = int(args.eval_freq * args.num_steps_per_epoch)

    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps + 1):
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        env.reset()
        loss_metric = agent.train(replay_buffer=data_sampler,
                                  batch_size=args.batch_size)
        training_iters += 1
        # Logging
        if training_iters % log_interval == 0:
            if loss_metric is not None:
                utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
                logger.record_tabular('Trained Epochs', curr_epoch)
                logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
                logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
                logger.record_tabular('Distill Loss', np.mean(loss_metric['distill_loss']))
                logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
                logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
                logger.record_tabular('Gamma Loss', np.mean(loss_metric['gamma_loss']))

                # Evaluating
                eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent,
                                                                                       args.env_name,
                                                                                       args.seed,
                                                                                       eval_episodes=args.eval_episodes)
                logger.record_tabular('Average Episodic Reward', eval_res)
                logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
                logger.record_tabular('Average Episodic N-Reward Std', eval_norm_res_std)
                logger.dump_tabular()

                if args.save_checkpoints:
                    agent.save_model(output_dir, curr_epoch)
    agent.save_model(output_dir, curr_epoch)


# Runs policy for [eval_episodes] episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument("--env_name", default="antmaze-large-diverse-v0", type=str, help='Mujoco Gym environment')
    parser.add_argument("--seed", default=1, type=int, help='random seed (default: 0)')
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--dir", default="results", type=str)

    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--repeats", default=1024, type=int)
    parser.add_argument("--tau", default=0.005, type=float)

    parser.add_argument("--sigma_max", default=80, type=int)
    parser.add_argument("--sigma_min", default=0.002, type=int)
    parser.add_argument("--sigma_data", default=0.5, type=int)
    parser.add_argument('--save_checkpoints', action='store_true')

    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--discount", default=0.99, type=float, help='discount factor for reward (default: 0.99)')

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    if 'antmaze' in args.env_name:
        args.reward_tune = 'iql_antmaze'
        args.eval_episodes = 100
    else:
        args.reward_tune = 'no'
        args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.num_epochs = offline_hyperparameters[args.env_name]["num_epochs"]
    args.lr = offline_hyperparameters[args.env_name]["lr"]
    args.lr_decay = offline_hyperparameters[args.env_name]["lr_decay"]
    args.batch_size = offline_hyperparameters[args.env_name]["batch_size"]
    args.alpha = offline_hyperparameters[args.env_name]["alpha"]
    args.gamma = offline_hyperparameters[args.env_name]["gamma"]
    args.expectile = offline_hyperparameters[args.env_name]["expectile"]

    file_name = f'|expect-{args.expectile}'
    file_name += f"|alpha-{args.alpha}|gamma-{args.gamma}"
    file_name += f'|seed={args.seed}'
    file_name += f'|lr={args.lr}'
    if args.lr_decay:
        file_name += f'|lr_decay'
    if args.pretrain_epochs is not None:
        file_name += f'|pretrain={args.pretrain_epochs}'
    #file_name += f'|{args.env_name}'

    results_dir = os.path.join(args.output_dir, args.env_name, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")

    variant = vars(args)
    variant.update(version=f"DTQL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                args.device,
                results_dir,
                args)
