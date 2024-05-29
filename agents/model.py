import torch
import torch.nn as nn

########################################### Critic net ################################
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.v_model = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                     nn.Mish(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Mish(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Mish(),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def v(self, state):
        return self.v_model(state)


############################################# SAC #################################################
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class DiagGaussianActorTanhAction(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_dim, action_dim, max_action,
                 hidden_dim=256, hidden_depth=3,
                 log_std_bounds=[-5, 2]):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.net = mlp(state_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.apply(weight_init)
        self.action_scale = max_action
        self.action_dim = action_dim

    def forward(self, state):
        mu, log_std = self.net(state).chunk(2, dim=-1)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std.clamp(log_std_min, log_std_max)

        std = log_std.exp()
        actor_dist = torch.distributions.Normal(mu, std)
        return actor_dist

    def sample(self, state):
        actor_dist = self(state)
        z = actor_dist.rsample()
        action = torch.tanh(z)

        action = action * self.action_scale
        action = action.clamp(-self.action_scale, self.action_scale)
        return action

    def log_prob(self, state, action):
        actor_dist = self(state)
        pre_tanh_value = torch.arctanh(action / (self.action_scale + 1e-3))
        log_prob = actor_dist.log_prob(pre_tanh_value)
        return log_prob.sum(-1, keepdim=True)

    def get_entropy(self,state):
        with torch.no_grad():
            mu, log_std = self.net(state).chunk(2, dim=-1)
        return log_std.sum(-1).mean()



