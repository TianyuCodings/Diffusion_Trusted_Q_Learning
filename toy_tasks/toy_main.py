import argparse
import os
import torch
from tqdm import tqdm
import sys
sys.path.append(("../"))

from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork
from data_generator import DataGenerator
from agents.model import DiagGaussianActorTanhAction, Critic
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import torch.nn.functional as F
import random


def q_v_critic_loss(critic, state, action, next_state, reward, not_done, expectile, discount):
    def expectile_loss(diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    with torch.no_grad():
        q = critic.q_min(state, action)
    v = critic.v(state)
    value_loss = expectile_loss(q - v, expectile).mean()

    current_q1, current_q2 = critic(state, action)
    with torch.no_grad():
        next_v = critic.v(next_state)
    target_q = (reward + not_done * discount * next_v).detach()

    critic_loss = value_loss + F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    return critic_loss

def get_dmd_loss(diffusion_model, true_score_model, fake_score_model, fake_action_data, state_data):
    noise = torch.randn_like(fake_action_data)
    with torch.no_grad():
        pred_real_action, _, t_chosen = diffusion_model.diffusion_train_step(model=true_score_model, x=fake_action_data, cond=state_data,
                                                                       noise=noise, t_chosen=None, return_denoised=True)

        pred_fake_action, _, t_chosen = diffusion_model.diffusion_train_step(model=fake_score_model, x=fake_action_data, cond=state_data,
                                                                       noise=noise, t_chosen=t_chosen,
                                                                       return_denoised=True)
        weighting_factor = (fake_action_data - pred_real_action).abs().mean(axis=1).reshape(-1, 1)
        grad = (pred_fake_action - pred_real_action) / weighting_factor
    distill_loss = 0.5 * F.mse_loss(fake_action_data, (fake_action_data - grad).detach())
    return distill_loss

def train_toy_task(args,file_name):
    data_manager = DataGenerator(args.env_name)
    action, state = data_manager.generate_samples(10000)
    device = args.device

    """Data Prepare"""
    state = state.to(torch.float32).to(device)
    state = state.reshape(-1, 1)
    next_state = state.reshape(-1, 1).to(device)
    reward = np.linalg.norm(action, axis=1, keepdims=True)

    if args.reward_type == "far":
        # farer tp (0,0), higher reward
        reward = torch.from_numpy(reward).to(torch.float32).to(device)
    elif args.reward_type == "near":
        # closer to (0,0), higher reward
        reward = torch.from_numpy(np.max(reward) - reward).to(torch.float32).to(device)
    elif args.reward_type == "hard": # reward type is hard
        reward[(action[:, 0] < -7.5) & (action[:, 1] > 7.5)] = 2 * reward[(action[:, 0] < -7.5) & (action[:, 1] > 7.5)]
        reward = torch.from_numpy(reward).to(torch.float32).to(device)
    elif args.reward_type == "same":
        reward = torch.from_numpy(np.zeros_like(reward)).to(torch.float32).to(device)

    action = action.to(torch.float32).to(device)
    not_done = torch.ones_like(reward).to(torch.float32)

    """Init bc actor"""
    bc_actor = ScoreNetwork(
        action_dim=2,
        hidden_dim=128,
        time_embed_dim=4,
        cond_dim=1,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=2,
        device=device,
        cond_conditional=True
    ).to(device)
    bc_actor_optimizer = torch.optim.Adam(bc_actor.parameters(), lr=3e-3)

    diffusion = DiffusionModel(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
    )

    if args.actor == "sac":
        distill_actor = DiagGaussianActorTanhAction(state_dim=1, action_dim=2,
                                              max_action=action.abs().max(),use_scale=True).to(device)
    elif args.actor == "implicit":
        distill_actor = ScoreNetwork(
            action_dim=2,
            hidden_dim=128,
            time_embed_dim=4,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=2,
            device=device,
            cond_conditional=True
        ).to(device)

        if args.pretrain_diffusion:
            for _ in tqdm(range(args.pretrain_epochs)):
                loss = diffusion.diffusion_train_step(bc_actor, action, state)
                bc_actor_optimizer.zero_grad()
                loss.backward()
                bc_actor_optimizer.step()
            bc_actor_state_dict = bc_actor.state_dict()
            distill_actor.load_state_dict(bc_actor_state_dict)

    else:
        raise ValueError("Actor type can only be sac for implicit")
    distill_actor_optimizer = torch.optim.Adam(distill_actor.parameters(), lr=3e-3)

    critic = Critic(state_dim=1, action_dim=2).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)

    if args.distill_loss == "dmd":
        distill_score = ScoreNetwork(
            action_dim=2,
            hidden_dim=128,
            time_embed_dim=4,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=2,
            device=device,
            cond_conditional=True
        ).to(device)
        distill_score_optimizer = torch.optim.Adam(distill_score.parameters(), lr=3e-3)

    def get_action(given_state, action_dim, generation_sigma=2.5):
        if args.actor == "sac":
            action = distill_actor.sample(state=given_state)
            return action
        elif args.actor == "implicit":
            noise = torch.randn((given_state.shape[0], action_dim)) * generation_sigma
            noise = noise.to(given_state.device)
            action = distill_actor(noise, given_state, torch.tensor([generation_sigma]).to(given_state.device))
            return action
        else:
            raise ValueError("Actor not correct.")

    pbar = tqdm(range(args.train_epochs))
    for i in range(args.train_epochs):
        """Q policy"""
        critic_loss = q_v_critic_loss(critic,state,action, next_state, reward, not_done, args.expectile, args.discount)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        """BC policy"""
        loss = diffusion.diffusion_train_step(bc_actor, action, state)
        bc_actor_optimizer.zero_grad()
        loss.backward()
        bc_actor_optimizer.step()

        """Distill policy"""
        new_action = get_action(given_state=state, action_dim=2,
                                         generation_sigma=args.generation_sigma)
        q_loss = -critic.q_min(state, new_action).mean()

        if args.distill_loss == "diffusion":
            distill_loss = diffusion.diffusion_train_step(bc_actor, new_action, state)
        elif args.distill_loss == "dmd":
            distill_loss = get_dmd_loss(diffusion, bc_actor, distill_score, new_action, state)
        else:
            distill_loss = 0

        if args.gamma == 0.:
            gamma_loss = 0
        else:
            gamma_loss = -distill_actor.log_prob(state,action).mean()

        actor_loss = distill_loss + args.eta * q_loss + args.gamma * gamma_loss
        distill_actor_optimizer.zero_grad()
        actor_loss.backward()
        distill_actor_optimizer.step()

        """Train fake score"""
        if args.distill_loss == "dmd":
            fake_loss = diffusion.diffusion_train_step(distill_score, new_action.detach(), state)
            distill_score_optimizer.zero_grad()
            fake_loss.backward()
            distill_score_optimizer.step()

        """Update critic target"""
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            tau = 0.005
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        pbar.set_description(f"Step {i}, BC Loss: {loss:.4f}, Critic Loss: {critic_loss:.4f}, Q loss:{q_loss:.4f}")
        pbar.update(1)


    state = torch.zeros(1000).int().reshape(-1, 1).to(device).to(torch.float32)
    with torch.no_grad():
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(action.cpu().numpy()[:, 0], action.cpu().numpy()[:, 1], c=reward.cpu().numpy())
        plt.colorbar(scatter, label='Reward values')
        plt.title(args.env_name)
        plt.xlabel('Action Dimension 1')
        plt.ylabel('Action Dimension 2')
        plt.grid(True)

        new_action = get_action(given_state=state,action_dim=2,
                                         generation_sigma=args.generation_sigma)
        new_action = new_action.cpu().numpy()
        plt.scatter(new_action[:, 0], new_action[:, 1], c='red',alpha=0.5)
        plot_path = file_name + ".png"
        plt.savefig(plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--actor', default="sac", type=str,help="sac or implicit")
    parser.add_argument('--distill_loss', default="dmd", type=str, help="diffusion or dmd")
    parser.add_argument("--env_name", default="25_gaussian", type=str, help="swiss_roll_2D or 25_gaussian")
    parser.add_argument("--pretrain_diffusion", action="store_true")
    parser.add_argument("--train_epochs", default=3500, type=int)
    parser.add_argument("--pretrain_epochs", default=500, type=int)
    parser.add_argument("--reward_type", default="hard", type=str, help="far, near, or hard")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--eta", default=1, type=float)
    parser.add_argument("--generation_sigma", default=2.5, type=float)
    parser.add_argument("--gamma", default=0.,type=float,help="weight of sac entropy")
    parser.add_argument("--expectile", default=0.95, type=float)

    parser.add_argument("--sigma_max", default=80, type=float)
    parser.add_argument("--sigma_min", default=0.002, type=float)
    parser.add_argument("--sigma_data", default=0.5, type=float)

    args = parser.parse_args()
    output_dir = f"results/{args.env_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"actor={args.actor}|distill={args.distill_loss}|seed={args.seed}|reward={args.reward_type}"
    file_name = os.path.join(output_dir,file_name)
    if args.actor == "implicit":
        file_name += f"|pretrain={args.pretrain_diffusion}"
    file_name += f"|eta={args.eta}"
    file_name += f"|gamma={args.gamma}"

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # May impact performance

    set_seed(args.seed)
    train_toy_task(args, file_name)