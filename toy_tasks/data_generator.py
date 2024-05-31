import numpy as np
import torch

class action_space:
    def __init__(self,high,low,action_dim):
        self.high = np.ones(action_dim)*high.item()
        self.low = np.ones(action_dim)*low.item()

class DataGenerator:
    def __init__(self, dist_type: str, reward_type: str, device):
        self.dist_type = dist_type
        self.reward_type = reward_type
        self.device = device
        if self.dist_type in {"swiss_roll_2D", "25_gaussian"}:
            pass
        else:
            raise ValueError("dist_type is not valid.")

    def generate_samples(self, num_samples: int):
        if self.dist_type == "swiss_roll_2D":
            action, state = self.sample_swiss_roll(num_samples)
        elif self.dist_type == "25_gaussian":
            action, state = self.sample_25_gaussian(num_samples)
        state = state.to(torch.float32).to(self.device)
        state = state.reshape(-1, 1)
        next_state = state.reshape(-1, 1).to(self.device)
        reward = np.linalg.norm(action, axis=1, keepdims=True)

        if self.reward_type == "far":
            # farer tp (0,0), higher reward
            reward = torch.from_numpy(reward).to(torch.float32).to(self.device)
        elif self.reward_type == "near":
            # closer to (0,0), higher reward
            reward = torch.from_numpy(np.max(reward) - reward).to(torch.float32).to(self.device)
        elif self.reward_type == "hard":  # reward type is hard
            reward[(action[:, 0] < -7.5) & (action[:, 1] > 7.5)] = 2 * reward[
                (action[:, 0] < -7.5) & (action[:, 1] > 7.5)]
            reward = torch.from_numpy(reward).to(torch.float32).to(self.device)
        elif self.reward_type == "same":
            reward = torch.from_numpy(np.zeros_like(reward)).to(torch.float32).to(self.device)

        action = action.to(torch.float32).to(self.device)
        not_done = torch.ones_like(reward).to(torch.float32)

        self.action = action
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.not_done = not_done

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        #normalized to [-1,1]
        #self.action /= self.action.abs().max()

        self.action_space = action_space(high=self.action.max(),low=self.action.min(),action_dim=self.action_dim)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
        )

    @staticmethod
    def sample_swiss_roll(num_samples):
        from sklearn.datasets import make_swiss_roll
        samples = make_swiss_roll(num_samples, noise=0.1)
        samples = torch.tensor(samples[0][:, [0, 2]], dtype=torch.float32)
        # None is the placeholder for label
        return samples, torch.zeros(num_samples).int()

    @staticmethod
    def sample_25_gaussian(num_samples):
        num_modes = 25  # Number of Gaussian modes
        grid_size = int(np.sqrt(num_modes))  # Determining the grid size (5x5 for 25 modes)

        # Creating a grid of means
        x_means = np.linspace(-10, 10, grid_size)
        y_means = np.linspace(-10, 10, grid_size)
        means = np.array(np.meshgrid(x_means, y_means)).T.reshape(-1, 2)

        # Standard deviation for each mode (can be adjusted as needed)
        # Standard deviation for each mode (can be adjusted as needed)
        std_dev = 0.3
        covariance_matrix = np.array([[std_dev ** 2, 0], [0, std_dev ** 2]])  # Diagonal covariance matrix

        # Generating one sample from each mode
        samples = np.array(
            [np.random.multivariate_normal(mean, covariance_matrix, num_samples // num_modes) for mean in means])
        samples = samples.reshape(-1, samples.shape[-1])
        samples = torch.from_numpy(samples).type(torch.float32)
        return samples, torch.zeros(num_samples).int()