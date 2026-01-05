import torch
import torch.nn as nn
import types
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import logging
from typing import List, Optional, Tuple

# 引入內部模組
from .modules.layer_controller import TransformerLayerController

logger = logging.getLogger(__name__)

def _custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    自定義的 Attention Forward 函數，用於替換 LlamaAttention.forward。
    將資料流導向 LiveKVQuant-P 的 TransformerLayerController。
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Linear Projection (取得 Q, K, V)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 2. Reshape
    # [修正] 直接從 config 讀取參數，避免 AttributeError
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.config.hidden_size // num_heads

    # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # 3. RoPE 計算 (Rotary Positional Embedding)
    # [相容性修正] 檢查 kwargs 是否已有 position_embeddings (新版 transformers)
    cos, sin = None, None
    if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
        cos, sin = kwargs["position_embeddings"]
    else:
        # 舊版或是手動計算 fallback
        # Llama 3 使用的是 self.rotary_emb (我們在 inject 時確保了它存在)
        kv_seq_len = key_states.shape[-2]
        
        if position_ids is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)

    # 4. RoPE 應用策略 (對應論文 Pre-RoPE Key Quantization)
    # Q: 必須套用 RoPE，因為 Controller/AttentionCore 需要用它來做 Dot Product
    query_states_rotated, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

    # 5. 呼叫 Controller
    # [修正] 直接從 self 取得 controller，避免透過 parent_layer 造成循環參照
    controller = self.livekv_controller
    
    # 執行 LiveKVQuant-P 流程: Stats -> Quantize -> Store -> Attention
    # 注意傳入的是: Rotated Q, Raw K, Raw V
    attn_output = controller(query_states_rotated, key_states, value_states)

    # 6. Reshape Output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)

    # 我們接管了 Cache，所以回傳 None 給 HF 以節省記憶體
    return attn_output, None


class LiveKVQuantModel:
    """
    LiveKVQuant-P 的模型包裝器。
    
    職責：
    1. 載入 HuggingFace 基礎模型 (Llama-3)。
    2. 使用 Monkey Patch 替換 Attention Forward。
    3. 實作 End-to-End 的 Chunk-wise Prefill + Decoding 流程。
    """

    def __init__(self, model_id: str, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        logger.info(f"Loading tokenizer and model from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 載入模型 (FP16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # [核心步驟] 注入 Controllers 並替換 Forward
        self._inject_controllers()
        
        # 設定為評估模式
        self.model.eval()

    def _inject_controllers(self):
        """
        為模型的每一層掛載 Controller，並執行 Monkey Patching。
        """
        logger.info("Injecting TransformerLayerControllers and patching LlamaAttention...")
        
        self.layers = self.model.model.layers
        self.controllers = []

        # [相容性修正] 嘗試獲取全域 RoPE 模組 (適用於新版 transformers)
        global_rotary_emb = None
        if hasattr(self.model.model, "rotary_emb"):
            global_rotary_emb = self.model.model.rotary_emb

        for i, layer in enumerate(self.layers):
            # 1. 初始化 Controller
            controller = TransformerLayerController(self.config, layer_idx=i)
            
            # [相容性修正] 處理 RoPE 模組綁定
            if hasattr(layer.self_attn, "rotary_emb"):
                controller.rotary_emb_module = layer.self_attn.rotary_emb
            elif global_rotary_emb is not None:
                controller.rotary_emb_module = global_rotary_emb
                # [關鍵] 將全域模組掛載回 self_attn，確保 _custom_attention_forward 的 fallback 邏輯能運作
                layer.self_attn.rotary_emb = global_rotary_emb
            else:
                logger.warning(f"Layer {i}: Could not find rotary_emb module!")

            # 2. 掛載 Controller
            # [修正] 將 Controller 直接綁定給 self_attn，避免循環參照
            layer.self_attn.livekv_controller = controller
            
            # 為了方便管理，我們也保留在 list 中
            self.controllers.append(controller)
            
            # 3. [已移除] 移除 parent_layer 的綁定，解決 RecursionError
            # layer.self_attn.parent_layer = layer 
            
            # 4. [Monkey Patch] 替換 forward 方法
            # 使用 types.MethodType 將 function 綁定為 instance method
            layer.self_attn.forward = types.MethodType(_custom_attention_forward, layer.self_attn)
            
        logger.info(f"Successfully patched {len(self.controllers)} layers.")

    def _chunk_input(self, input_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        seq_len = input_ids.size(1)
        chunks = []
        for i in range(0, seq_len, chunk_size):
            chunks.append(input_ids[:, i : i + chunk_size])
        return chunks

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        執行完整的推論流程：Prefill (Chunk-wise) -> Decoding。
        """
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        seq_len = input_ids.size(1)
        
        # logger.info(f"Processing Prompt: {seq_len} tokens")

        # 2. Chunking
        chunks = self._chunk_input(input_ids, self.config.chunk_size)
        # logger.info(f"Split into {len(chunks)} chunks (Size: {self.config.chunk_size})")

        # 重置 Controller 狀態
        for controller in self.controllers:
            controller.reset_cache()

        # 3. Prefill Phase (One by one)
        current_pos = 0
        
        # logger.info("Starting Prefill Phase...")
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                chunk_len = chunk.size(1)
                
                # 設定 Chunk Index (用於 Warm-up 判斷)
                for controller in self.controllers:
                    controller.set_chunk_idx(i)

                # [關鍵] 手動計算 Position IDs，確保 RoPE 在分塊處理時位置正確
                position_ids = torch.arange(current_pos, current_pos + chunk_len, device=self.device).unsqueeze(0)
                
                # Forward pass
                # 我們傳入 use_cache=False，因為我們有自己的 Cache 機制
                self.model(
                    input_ids=chunk,
                    position_ids=position_ids,
                    use_cache=False 
                )
                
                current_pos += chunk_len

        # logger.info("Prefill Completed. Starting Decoding Phase...")

        # 4. Decoding Phase
        # 準備最後一個 token 進行生成
        next_token = input_ids[:, -1:]
        generated_ids = []
        
        # 通知 Controller 進入 Decoding 模式
        for controller in self.controllers:
            controller.set_decoding_mode()

        for _ in range(max_new_tokens):
            position_ids = torch.tensor([[current_pos]], device=self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=next_token,
                position_ids=position_ids,
                use_cache=False
            )
            
            # 取樣
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            generated_ids.append(next_token.item())
            current_pos += 1
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # 5. Decode Output
        # 只解碼生成的 id
        generated_tensor = torch.tensor(generated_ids, device=self.device)
        output_text = self.tokenizer.decode(generated_tensor, skip_special_tokens=True)
        
        return output_text