import torch
import pytest
from livekvquant.utils.ema_utils import update_clipped_ema

def test_ema_initialization():
    """
    測試 Warm-up 階段的初始行為。
    當沒有歷史紀錄 (prev_ema is None) 時，應直接回傳當前的 Absmax。
    """
    current_absmax = torch.tensor([1.0])
    mu_t = update_clipped_ema(current_absmax, prev_ema=None, alpha=0.1, clip_n=1.5)
    assert torch.allclose(mu_t, current_absmax), "Initial EMA should equal current absmax"

def test_ema_normal_update():
    """
    測試正常的 EMA 更新 (無 Clipping)。
    公式: mu_t = (1-alpha)*prev + alpha*curr
    """
    prev_ema = torch.tensor([10.0])
    current_absmax = torch.tensor([12.0]) # 12 < 1.5 * 10 (No clipping)
    alpha = 0.5
    clip_n = 1.5
    
    # Expected: 0.5 * 10 + 0.5 * 12 = 11.0
    expected = 11.0
    mu_t = update_clipped_ema(current_absmax, prev_ema, alpha, clip_n)
    
    assert torch.allclose(mu_t, torch.tensor([expected]))

def test_ema_clipping_logic():
    """
    [核心測試] 驗證 Clipped Update 是否生效。
    對應論文 Eq 3-3: v_t = min(m_t, N * mu_{t-1})
    """
    prev_ema = torch.tensor([10.0])
    spike_absmax = torch.tensor([100.0]) # 巨大的突波 (Spike)
    alpha = 0.5
    clip_n = 1.5 # 允許最大增長為 1.5 倍 => limit = 15.0
    
    # Expected: 
    # v_t = min(100, 15) = 15
    # mu_t = 0.5 * 10 + 0.5 * 15 = 12.5 (而不是 55.0)
    expected = 12.5
    
    mu_t = update_clipped_ema(spike_absmax, prev_ema, alpha, clip_n)
    
    assert torch.allclose(mu_t, torch.tensor([expected])), "Spike should be clipped!"