import copy
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from pathlib import Path
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork
from agents.model import Critic
from agents.helpers import EMA, get_dmd_loss
import numpy as np
class DQL_KL(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 action_space=None,
                 discount=0.99,
                 alpha=1.0,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 sigma_max=80.,
                 sigma_min=0.002,
                 sigma_data=0.5,
                 generation_sigma=2.5,
                 expectile=0.7,
                 tau = 0.005,
                 gamma=0,
                 repeats=1024
                 ):
        """Init critic networks"""
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        """"Init behaviour cloning network"""
        self.bc_actor = ScoreNetwork(
            action_dim=action_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=state_dim,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=action_dim,
            device=device,
            cond_conditional=True
        ).to(device)
        self.bc_actor_target = copy.deepcopy(self.bc_actor)
        self.bc_actor_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=lr)

        """Init diffusion schedule"""
        self.diffusion = DiffusionModel(
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
            clip_denoised=True,
            max_action=float(action_space.high[0]))

        """Init one-step policy"""
        self.distill_actor = ScoreNetwork(
            action_dim=action_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=state_dim,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=action_dim,
            device=device,
            cond_conditional=True
        ).to(device)
        self.distill_actor_target = copy.deepcopy(self.distill_actor)
        self.distill_actor_optimizer = torch.optim.Adam(self.distill_actor.parameters(), lr=lr)

        """Init fake score network"""
        self.fake_score = ScoreNetwork(
            action_dim=action_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=state_dim,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=action_dim,
            device=device,
            cond_conditional=True
        ).to(device)
        self.fake_score_optimizer = torch.optim.Adam(self.fake_score.parameters(), lr=lr)

        """Back up training parameters"""
        self.generation_sigma = generation_sigma
        self.tau = tau
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.repeats = repeats

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.update_ema_every = update_ema_every

        if lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
            self.bc_actor_lr_scheduler = CosineAnnealingLR(self.bc_actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.distill_actor_lr_scheduler = CosineAnnealingLR(self.distill_actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.fake_score_lr_scheduler = CosineAnnealingLR(self.fake_score_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.alpha = alpha  # bc weight
        self.expectile = expectile
        self.device = device

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.distill_actor_target, self.distill_actor)

    def pretrain(self,replay_buffer, batch_size=256,pretrain_steps=50000):
        for _ in range(pretrain_steps):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            loss = self.diffusion.diffusion_train_step(self.bc_actor, action, state)
            self.bc_actor_optimizer.zero_grad()
            loss.backward()
            self.bc_actor_optimizer.step()
            self.bc_loss = loss

            critic_loss = self.q_v_critic_loss(state,action, next_state, reward, not_done)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        bc_actor_state_dict = self.bc_actor.state_dict()
        self.distill_actor.load_state_dict(bc_actor_state_dict)
        self.fake_score.load_state_dict(bc_actor_state_dict)
    def load_or_pretrain_models(self, dir, replay_buffer, batch_size, pretrain_steps,num_steps_per_epoch):
        # Paths for the models
        actor_path = Path(dir) / f'diffusion_pretrained_{pretrain_steps // num_steps_per_epoch}.pth'
        critic_path = Path(dir) / f'critic_pretrained_{pretrain_steps // num_steps_per_epoch}.pth'

        # Check if both models exist
        if actor_path.exists() and critic_path.exists():
            try:
                # Load the models
                self.bc_actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                bc_actor_state_dict = self.bc_actor.state_dict()
                self.distill_actor.load_state_dict(bc_actor_state_dict)
                self.fake_score.load_state_dict(torch.load(actor_path))
            except Exception as e:
                print(f"Failed to load models: {e}")
        else:
            # Begin pretraining if the models do not exist
            print("Models not found, starting pretraining...")
            self.pretrain(replay_buffer, batch_size, pretrain_steps)
            torch.save(self.bc_actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            print(f"Saved successfully to {dir}")

    def train(self, replay_buffer, batch_size=256):
        # initialize
        self.bc_loss = torch.tensor([0.]).to(self.device)
        self.critic_loss = torch.tensor([0.]).to(self.device)
        metric = {'bc_loss': [], 'distill_loss':[], 'ql_loss': [], 'actor_loss': [], 'critic_loss': [], 'gamma_loss': []}
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        """ Q Training """
        critic_loss = self.q_v_critic_loss(state, action, next_state, reward, not_done)

        self.critic_loss = critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """ Diffusion Policy Training """
        bc_loss = self.diffusion.diffusion_train_step(self.bc_actor, action, state)
        self.bc_actor_optimizer.zero_grad()
        bc_loss.backward()
        self.bc_actor_optimizer.step()
        self.bc_loss = bc_loss

        """Distill Policy Training"""
        new_action = self.get_agent_sample(self.distill_actor, given_state=state,
                                            generation_sigma=self.generation_sigma)
        distill_loss = get_dmd_loss(self.diffusion,self.bc_actor,self.fake_score,new_action,state)
        q_loss = -self.critic.q_min(state, new_action).mean()

        actor_loss = self.alpha * distill_loss + q_loss
        self.distill_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.distill_actor_optimizer.step()

        """Training fake score"""
        fake_score_loss = self.diffusion.diffusion_train_step(self.fake_score, new_action.detach(), state)
        self.fake_score_optimizer.zero_grad()
        fake_score_loss.backward()
        self.fake_score_optimizer.step()

        """ Step Target network """
        if self.step % self.update_ema_every == 0:
            self.step_ema()
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1

        metric['actor_loss'].append(actor_loss.item())
        metric['bc_loss'].append(self.bc_loss.item())
        metric['ql_loss'].append(q_loss.item())
        metric['critic_loss'].append(self.critic_loss.item())
        metric['distill_loss'].append(distill_loss.item())
        metric['gamma_loss'].append(np.nan)

        if self.lr_decay:
            self.bc_actor_lr_scheduler.step()
            self.distill_actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.fake_score_lr_scheduler.step()
        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=self.repeats, dim=0)
        with torch.no_grad():
            action = self.get_agent_sample(self.distill_actor, given_state=state_rpt,
                                  generation_sigma=self.generation_sigma
                                  )
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
        idx = torch.multinomial(F.softmax(q_value), 1)
        action = action[idx].cpu().data.numpy().flatten()
        return action

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.bc_actor.state_dict(), f'{dir}/bc_actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
            torch.save(self.distill_actor.state_dict(), f'{dir}/distill_actor_{id}.pth')
            torch.save(self.fake_score.state_dict(), f'{dir}/fake_score_{id}.pth')
        else:
            torch.save(self.bc_actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
            torch.save(self.distill_actor.state_dict(), f'{dir}/distill_actor.pth')
            torch.save(self.fake_score.state_dict(), f'{dir}/fake_score.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.bc_actor.load_state_dict(torch.load(f'{dir}/bc_actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
            self.distill_actor.load_state_dict(torch.load(f'{dir}/distill_actor_{id}.pth'))
            self.fake_score.load_state_dict(torch.load(f'{dir}/fake_score_{id}.pth'))
        else:
            self.bc_actor.load_state_dict(torch.load(f'{dir}/bc_actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
            self.distill_actor.load_state_dict(torch.load(f'{dir}/distill_actor.pth'))
            self.fake_score.load_state_dict(torch.load(f'{dir}/fake_score.pth'))
        print(f"Models loaded successfully from {dir}")


    def get_agent_sample(self,model,given_state,generation_sigma):
        noise = torch.randn((given_state.shape[0],self.action_dim)).to(given_state.device) * generation_sigma
        action = model(noise, given_state, torch.tensor([generation_sigma]).to(given_state.device))
        return action

    def q_v_critic_loss(self,state,action, next_state, reward, not_done):
        def expectile_loss(diff, expectile=0.8):
            weight = torch.where(diff > 0, expectile, (1 - expectile))
            return weight * (diff ** 2)

        with torch.no_grad():
            q = self.critic.q_min(state, action)
        v = self.critic.v(state)
        value_loss = expectile_loss(q - v, self.expectile).mean()

        current_q1, current_q2 = self.critic(state, action)
        with torch.no_grad():
            next_v = self.critic.v(next_state)
        target_q = (reward + not_done * self.discount * next_v).detach()

        critic_loss = value_loss + F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        return critic_loss
