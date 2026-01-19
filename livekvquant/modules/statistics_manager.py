import torch
from ..utils.ema_utils import update_clipped_ema

class StatisticsManager:
    def __init__(self, config):
        self.alpha = config.ema_alpha
        self.clip_n = config.clip_factor_n
        self.mu_t_prev = None 
        # [還原重點] 舊版預設有 History
        self.history = []

    def reset(self):
        self.mu_t_prev = None
        self.history = []

    def update_key_stats(self, k_tensor: torch.Tensor) -> torch.Tensor:
        # Per-channel EMA Absmax
        m_t = torch.amax(torch.abs(k_tensor), dim=-2, keepdim=True) 
        mu_t = update_clipped_ema(m_t, self.mu_t_prev, self.alpha, self.clip_n)
        
        # [還原重點] 包含這個同步操作 (雖然慢，但這是你之前的版本)
        self.history.append((m_t.detach().cpu(), mu_t.detach().cpu()))
        
        self.mu_t_prev = mu_t
        return mu_t

    def get_value_stats(self, v_tensor: torch.Tensor) -> torch.Tensor:
        return torch.amax(torch.abs(v_tensor), dim=-1, keepdim=True)