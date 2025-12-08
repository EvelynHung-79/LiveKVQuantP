import torch
from ..utils.ema_utils import update_clipped_ema

class StatisticsManager:
    """
    負責計算與維護量化所需的 Scale。
    [Modified] 加入了 History Tracking 功能以進行視覺化分析。
    """
    def __init__(self, config):
        self.alpha = config.ema_alpha
        self.clip_n = config.clip_factor_n
        
        self.mu_t_prev = None 
        
        # [NEW] 用來儲存每個 Chunk 的統計數據
        # 格式: list of (raw_absmax, ema_scale) tuples
        self.history = []

    def reset(self):
        self.mu_t_prev = None
        # [NEW] 重置歷史紀錄
        self.history = []

    def update_key_stats(self, k_tensor: torch.Tensor) -> torch.Tensor:
        """
        Key (Pre-RoPE): Per-channel EMA Absmax tracking.
        """
        # 1. 計算當前 Chunk 的 Absmax (m_t)
        # [batch, num_heads, 1, head_dim]
        m_t = torch.amax(torch.abs(k_tensor), dim=-2, keepdim=True) 

        # 2. 執行 Clipped EMA Update
        mu_t = update_clipped_ema(m_t, self.mu_t_prev, self.alpha, self.clip_n)
        
        # [NEW] 記錄數據 (Detach 並轉到 CPU 以節省顯卡記憶體)
        # 我們存下 m_t (當下真實最大值) 和 mu_t (穩定後的 Scale)
        self.history.append((
            m_t.detach().cpu(), 
            mu_t.detach().cpu()
        ))
        
        # 更新狀態
        self.mu_t_prev = mu_t
        return mu_t

    def get_value_stats(self, v_tensor: torch.Tensor) -> torch.Tensor:
        # Value 不需要 EMA，所以這裡不用改
        return torch.amax(torch.abs(v_tensor), dim=-1, keepdim=True)