from dataclasses import dataclass

@dataclass
class LiveKVQuantConfig:
    """
    LiveKVQuant-P 量化與 Prefill 階段配置參數
    
    Attributes:
        chunk_size (int): 預填階段的分塊大小 (Chunking Size)，默認 512。
        n_warmup (int): Warm-up 階段所用的 chunk 數量，默認為 2。
        bits (int): 量化位元數，目前設為 4 表示 INT4。
        ema_alpha (float): EMA 指數移動平均的平滑係數 alpha，默認 0.1。
        clip_factor_n (float): Clipped EMA 裁剪因子 N，用於限制EMA值波動，默認 1.5。
        outlier_ratio (float): 保留下來的異常值比例 (top-k outliers)，目前設定為 0.01 (1%)。
    """
    
    chunk_size: int = 512            # 分塊大小，用於分塊處理 KV Cache
    n_warmup: int = 2                # Warm-up 階段 chunk 數量以穩定EMA統計
    bits: int = 4                    # 量化位元深度，INT4
    ema_alpha: float = 0.4          # EMA 平滑因子 α
    clip_factor_n: float = 1.5       # EMA 裁剪因子 N，避免過大波動
    outlier_ratio: float = 0.01      # 頂尖 1% 異常值保留為 FP16
    baseline: bool = False           # 若為 True，則強制不進行量化，全程使用 FP16 (視為 Warm-up 處理)
    
# 可以實例化配置對象直接使用
config = LiveKVQuantConfig()
