import torch
import torch.nn.functional as F
class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_dmd_loss(diffusion_model, true_score_model, fake_score_model, fake_action_data, state_data):
    noise = torch.randn_like(fake_action_data)
    fake_score_model.eval()
    true_score_model.eval()
    with torch.no_grad():
        pred_real_action, _, t_chosen = diffusion_model.diffusion_train_step(model=true_score_model,
                                                                             x=fake_action_data, cond=state_data,
                                                                             noise=noise, t_chosen=None,
                                                                             return_denoised=True)

        pred_fake_action, _, t_chosen = diffusion_model.diffusion_train_step(model=fake_score_model,
                                                                             x=fake_action_data, cond=state_data,
                                                                             noise=noise, t_chosen=t_chosen,
                                                                             return_denoised=True)
        weighting_factor = (fake_action_data - pred_real_action).abs().mean(axis=1).reshape(-1, 1)
        grad = (pred_fake_action - pred_real_action) / weighting_factor
    distill_loss = 0.5 * F.mse_loss(fake_action_data, (fake_action_data - grad).detach())
    return distill_loss