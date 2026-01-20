import torch
from collections import deque
from ..utils.ema_utils import update_clipped_ema

class StatisticsManager:
    def __init__(self, config):
        self.alpha = config.ema_alpha
        self.clip_n = config.clip_factor_n
        self.mu_t_prev = None 
        self.history = deque(maxlen=100)

    def reset(self):
        self.mu_t_prev = None
        self.history = []

    def update_key_stats(self, k_tensor: torch.Tensor) -> torch.Tensor:
        # Per-channel EMA Absmax
        m_t = torch.amax(torch.abs(k_tensor), dim=-2, keepdim=True) 
        mu_t = update_clipped_ema(m_t, self.mu_t_prev, self.alpha, self.clip_n)
        
        # 為了避免拖慢速度，只有在真正需要 debug 時才建議開啟這行
        # self.history.append((m_t.detach().cpu(), mu_t.detach().cpu()))
        
        self.mu_t_prev = mu_t
        return mu_t

    def get_value_stats(self, v_tensor: torch.Tensor) -> torch.Tensor:
        return torch.amax(torch.abs(v_tensor), dim=-1, keepdim=True)