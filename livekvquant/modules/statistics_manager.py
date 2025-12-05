import torch
from ..utils.ema_utils import update_clipped_ema

class StatisticsManager:
    """
    負責計算與維護量化所需的 Scale。
    對應論文 3.3 Statistics Manager Module.
    """
    def __init__(self, config):
        self.alpha = config.ema_alpha
        self.clip_n = config.clip_factor_n
        
        # 紀錄 Key 的 EMA Scale (Per-channel)
        # 由於不同 Layer/Head 維度可能不同，初始化為 None
        self.mu_t_prev = None 

    def reset(self):
        self.mu_t_prev = None

    def update_key_stats(self, k_tensor: torch.Tensor) -> torch.Tensor:
        """
        Key (Pre-RoPE): Per-channel EMA Absmax tracking.
        對應論文 3.3.2 Scale Stabilization.
        
        Args:
            k_tensor: [batch, num_heads, seq_len, head_dim]
        Returns:
            mu_t: [batch, num_heads, 1, head_dim] (Per-channel scale)
        """
        # 1. 計算當前 Chunk 的 Absmax (m_t)
        # Channel 是最後一個維度，所以要對 Sequence 維度 (dim=-2) 取 Max
        m_t = torch.amax(torch.abs(k_tensor), dim=-2, keepdim=True) 

        # 2. 執行 Clipped EMA Update (呼叫 utils)
        # 對應 Eq 3-3, 3-4
        mu_t = update_clipped_ema(m_t, self.mu_t_prev, self.alpha, self.clip_n)
        
        # 更新狀態
        self.mu_t_prev = mu_t
        return mu_t

    def get_value_stats(self, v_tensor: torch.Tensor) -> torch.Tensor:
        """
        Value (Post-RoPE): Per-token instantaneous Absmax.
        不使用 EMA，直接回傳當前 token 的最大值。
        
        Args:
            v_tensor: [batch, num_heads, seq_len, head_dim]
        Returns:
            scale: [batch, num_heads, seq_len, 1] (Per-token scale)
        """
        # Per-token: 對 head_dim (dim=-1) 取 Max
        return torch.amax(torch.abs(v_tensor), dim=-1, keepdim=True)