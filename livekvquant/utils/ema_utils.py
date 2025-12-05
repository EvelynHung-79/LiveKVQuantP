import torch

def update_clipped_ema(current_absmax: torch.Tensor, 
                       prev_ema: torch.Tensor, 
                       alpha: float, 
                       clip_n: float) -> torch.Tensor:
    """
    執行 Clipped Exponential Moving Average (EMA) 更新。
    對應論文 Chapter 3.3.2 Scale Stabilization.

    Args:
        current_absmax (m_t): 當前 Chunk 的最大絕對值 (Eq 3-2)。
        prev_ema (mu_{t-1}): 上一時刻的 EMA Scale。
        alpha (α): 平滑係數 (Smoothing Factor)。
        clip_n (N): 裁剪因子 (Clipping Factor)。

    Returns:
        mu_t: 更新後的 EMA Scale。
    """
    
    # 如果是第一個 Chunk (prev_ema 為空)，直接回傳當前值作為初始 Scale
    # 對應論文 3.2.2 Warm-up Phase 的概念：初期需要快速收斂
    if prev_ema is None:
        return current_absmax

    # --- Eq 3-3: Clipped update value (v_t) ---
    # v_t = min(m_t, N * mu_{t-1})
    # 防止當前 Chunk 的突波 (Spike) 過度影響歷史平均
    limit = clip_n * prev_ema
    v_t = torch.minimum(current_absmax, limit)

    # --- Eq 3-4: EMA update (mu_t) ---
    # mu_t = (1 - alpha) * mu_{t-1} + alpha * v_t
    mu_t = (1.0 - alpha) * prev_ema + alpha * v_t
    
    return mu_t