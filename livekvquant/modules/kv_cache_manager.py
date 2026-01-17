import torch

class KVCacheManager:
    def __init__(self, config, max_seq_len=4096, max_batch_size=1):
        self.config = config
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.current_len = 0
        
        # Robust Config Loading
        hf_config = getattr(config, "hf_config", config)
        self.num_heads = getattr(hf_config, "num_attention_heads", 32)
        self.hidden_size = getattr(hf_config, "hidden_size", 4096)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(hf_config, "num_key_value_heads", self.num_heads)
        if self.num_kv_heads is None: 
            self.num_kv_heads = self.num_heads

        # ==========================================
        # 1. Int8 Cache (Quantized Layers)
        # ==========================================
        self.k_cache = torch.zeros(
            (max_batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
            dtype=torch.int8, device="cuda"
        )
        
        # [FIX 1] Change k_scales shape: 1 -> max_seq_len (Support Per-Token Scale)
        self.k_scales = torch.zeros(
            (max_batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
            dtype=torch.bfloat16, device="cuda"
        )
        
        self.v_cache = torch.zeros(
            (max_batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
            dtype=torch.int8, device="cuda"
        )
        self.v_scales = torch.zeros(
            (max_batch_size, self.num_kv_heads, max_seq_len, 1),
            dtype=torch.bfloat16, device="cuda"
        )
        
        # ==========================================
        # 2. FP16 Cache (Warmup / Bypass Layers)
        # ==========================================
        self.warmup_k = []
        self.warmup_v = []
        self.is_warmup_mode = False

    def store_chunk(self, k_data, v_data):
        """
        統一入口：處理 LayerController 傳來的 dict
        """
        data_type = k_data.get("type", "quantized")
        
        if data_type == "warmup":
            # Warmup Mode: 存入 List
            self.is_warmup_mode = True
            self.warmup_k.append(k_data["data"])
            self.warmup_v.append(v_data["data"])
            self.current_len += k_data["data"].shape[-2]
            
        else:
            # Quantized Mode: 解包並存入 Int8 Tensor
            self.is_warmup_mode = False
            k_quant = k_data["quantized_data"]
            k_scale = k_data["scale"]
            v_quant = v_data["quantized_data"]
            v_scale = v_data["scale"]
            
            self.store_tokens(k_quant, k_scale, v_quant, v_scale)

    def store_tokens(self, k_quant, k_scale, v_quant, v_scale, valid_len=None):
        batch_size = k_quant.shape[0]
        seq_len_to_add = k_quant.shape[2] 
        start_pos = self.current_len
        end_pos = start_pos + seq_len_to_add
        
        if end_pos > self.max_seq_len:
            pass 

        # 寫入 Int8 Tensor
        eff_end = min(end_pos, self.max_seq_len)
        eff_len = eff_end - start_pos
        
        if eff_len > 0:
            # K Cache
            self.k_cache[:, :, start_pos:eff_end, :] = k_quant[:, :, :eff_len, :]
            
            # [FIX 2] Store k_scales slice-by-slice (Dynamic update)
            self.k_scales[:, :, start_pos:eff_end, :] = k_scale[:, :, :eff_len, :]
            
            # V Cache
            self.v_cache[:, :, start_pos:eff_end, :] = v_quant[:, :, :eff_len, :]
            self.v_scales[:, :, start_pos:eff_end, :] = v_scale[:, :, :eff_len, :]
            
            self.current_len = eff_end

    def get_views(self):
        """
        回傳 Cache View。
        """
        if self.is_warmup_mode:
            return self.warmup_k, None, self.warmup_v, None
        else:
            # [FIX 3] Return sliced view for k_scales
            return (
                self.k_cache[:, :, :self.current_len, :],
                self.k_scales[:, :, :self.current_len, :], 
                self.v_cache[:, :, :self.current_len, :],
                self.v_scales[:, :, :self.current_len, :]
            )

    def clear(self):
        self.current_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.warmup_k = []
        self.warmup_v = []
        self.is_warmup_mode = False