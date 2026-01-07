import torch
import torch.nn as nn
import types
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import logging
from typing import List, Optional, Tuple

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
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.config.hidden_size // num_heads

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # 3. RoPE 計算
    cos, sin = None, None
    if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
        cos, sin = kwargs["position_embeddings"]
    else:
        kv_seq_len = key_states.shape[-2]
        if position_ids is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)

    # 4. RoPE 應用策略
    query_states_rotated, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

    # 5. 呼叫 Controller
    controller = self.livekv_controller
    attn_output = controller(query_states_rotated, key_states, value_states)

    # 6. Reshape Output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


class LiveKVQuantModel:
    """
    LiveKVQuant-P 的模型包裝器。
    """

    def __init__(self, model_id: str, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        logger.info(f"Loading tokenizer and model from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # [CRITICAL FIX] 修改為 bfloat16 以解決 Llama-3 長文本精度問題
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self._inject_controllers()
        self.model.eval()

    def _inject_controllers(self):
        logger.info("Injecting TransformerLayerControllers and patching LlamaAttention...")
        
        self.layers = self.model.model.layers
        self.controllers = []

        global_rotary_emb = None
        if hasattr(self.model.model, "rotary_emb"):
            global_rotary_emb = self.model.model.rotary_emb

        for i, layer in enumerate(self.layers):
            controller = TransformerLayerController(self.config, layer_idx=i)
            
            if hasattr(layer.self_attn, "rotary_emb"):
                controller.rotary_emb_module = layer.self_attn.rotary_emb
            elif global_rotary_emb is not None:
                controller.rotary_emb_module = global_rotary_emb
                layer.self_attn.rotary_emb = global_rotary_emb
            else:
                logger.warning(f"Layer {i}: Could not find rotary_emb module!")

            layer.self_attn.livekv_controller = controller
            self.controllers.append(controller)
            
            # Monkey Patch
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
        [FIX] 修正重複處理最後一個 Token 的問題。
        """
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        seq_len = input_ids.size(1)

        # 2. Chunking
        chunks = self._chunk_input(input_ids, self.config.chunk_size)

        # 重置 Controller 狀態
        for controller in self.controllers:
            controller.reset_cache()

        # 3. Prefill Phase
        current_pos = 0
        last_chunk_logits = None # 用來存最後一個 chunk 的輸出
        
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                chunk_len = chunk.size(1)
                
                for controller in self.controllers:
                    controller.set_chunk_idx(i)

                position_ids = torch.arange(current_pos, current_pos + chunk_len, device=self.device).unsqueeze(0)
                
                # 執行 Forward，並捕捉輸出 (為了拿到 logits)
                outputs = self.model(
                    input_ids=chunk,
                    position_ids=position_ids,
                    use_cache=False 
                )
                
                current_pos += chunk_len
                
                # 如果是最後一個 chunk，我們需要它的 logits 來預測第一個新 token
                if i == len(chunks) - 1:
                    last_chunk_logits = outputs.logits

        # 4. Decoding Phase
        # [FIX] 使用 Prefill 最後計算出的 Logits 來預測第一個 token
        next_token_logits = last_chunk_logits[:, -1, :]
        
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
        generated_ids = [next_token.item()]
        
        # 通知 Controller 進入 Decoding 模式
        for controller in self.controllers:
            controller.set_decoding_mode()

        # [FIX] 迴圈少跑一次，因為我們已經產生第一個 token 了
        for _ in range(max_new_tokens - 1):
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
        generated_tensor = torch.tensor(generated_ids, device=self.device)
        output_text = self.tokenizer.decode(generated_tensor, skip_special_tokens=True)
        
        return output_text