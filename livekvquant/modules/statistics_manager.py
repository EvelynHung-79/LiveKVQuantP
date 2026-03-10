import torch
from collections import deque
from ..utils.ema_utils import update_clipped_ema

class StatisticsManager:
    def __init__(self, config):
        self.alpha = config.ema_alpha
        self.clip_n = config.clip_factor_n
        self.stats_method = config.stats_method  # "ema_absmax" or "ema_minmax"
        self.mu_t_prev = None     # EMA Absmax 用
        self.mu_max_prev = None   # EMA MinMax 用（正方向）
        self.mu_min_prev = None   # EMA MinMax 用（負方向，追蹤 |min|）
        self.history = deque(maxlen=100)

    def reset(self):
        self.mu_t_prev = None
        self.mu_max_prev = None
        self.mu_min_prev = None
        self.history = []

    def update_key_stats(self, k_tensor: torch.Tensor):
        """
        回傳格式依 stats_method 不同：
          - ema_absmax: 回傳 absmax tensor（供 symmetric quantization）
          - ema_minmax: 回傳 (ema_max, ema_min) tuple（供 asymmetric quantization）
            ema_max >= 0，ema_min <= 0
        """
        if self.stats_method == "ema_minmax":
            # Asymmetric: 分別追蹤正方向 max 和負方向 |min|，各做 Clipped EMA
            m_max = torch.amax(k_tensor, dim=-2, keepdim=True).clamp(min=0)
            m_min_abs = torch.amin(k_tensor, dim=-2, keepdim=True).abs()
            mu_max = update_clipped_ema(m_max, self.mu_max_prev, self.alpha, self.clip_n)
            mu_min = update_clipped_ema(m_min_abs, self.mu_min_prev, self.alpha, self.clip_n)
            self.mu_max_prev = mu_max
            self.mu_min_prev = mu_min
            # 回傳帶符號的 (max, min)，max >= 0, min <= 0
            return mu_max, -mu_min
        else:
            # Symmetric: Per-channel EMA Absmax（原有邏輯）
            m_t = torch.amax(torch.abs(k_tensor), dim=-2, keepdim=True)
            mu_t = update_clipped_ema(m_t, self.mu_t_prev, self.alpha, self.clip_n)
            self.mu_t_prev = mu_t
            return mu_t

    def get_value_stats(self, v_tensor: torch.Tensor):
        """
        V 的統計同樣依 stats_method 切換。
        注意：V 的 outlier dim 是 -1（channel），所以 reduce 在 -1。
        """
        if self.stats_method == "ema_minmax":
            v_max = torch.amax(v_tensor, dim=-1, keepdim=True).clamp(min=0)
            v_min = torch.amin(v_tensor, dim=-1, keepdim=True)  # 帶符號，<= 0
            return v_max, v_min
        else:
            return torch.amax(torch.abs(v_tensor), dim=-1, keepdim=True)
