import torch
import torch.nn as nn
import types
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List, Optional, Tuple

from .modules.layer_controller import TransformerLayerController

logger = logging.getLogger(__name__)

def _custom_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, use_cache=False, **kwargs):
    """
    攔截 HuggingFace Attention Forward，將 Q/K/V 導向 LayerController 進行量化與管理。
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.config.hidden_size // num_heads
    
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # 呼叫我們的 Controller
    controller = self.livekv_controller
    attn_output = controller(query_states, key_states, value_states, position_ids=position_ids)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None

class LiveKVQuantModel:
    def __init__(self, model_id: str, config, device: str = "cuda"):
        self.config = config
        self.device = device
        logger.info(f"Loading tokenizer and model from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self._inject_controllers()
        self.model.eval()

    def _inject_controllers(self):
        """
        將 LayerController 注入到模型的每一層 Attention 中，並替換 Forward 方法。
        """
        self.layers = self.model.model.layers
        self.controllers = []
        
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads
        if num_kv_heads is None: num_kv_heads = self.model.config.num_attention_heads
        
        self.config.head_dim = head_dim
        self.config.num_key_value_heads = num_kv_heads
        
        # 1. Patch Global RoPE (if exists)
        global_rotary_emb = None
        if hasattr(self.model.model, "rotary_emb"):
            global_rotary_emb = self.model.model.rotary_emb

        # 2. Inject per layer
        for i, layer in enumerate(self.layers):
            controller = TransformerLayerController(self.config, layer_idx=i)
            
            # Handle Layer-specific RoPE
            rope_module = None
            if hasattr(layer.self_attn, "rotary_emb"):
                rope_module = layer.self_attn.rotary_emb
            
            # Link RoPE to Controller
            if rope_module is None and global_rotary_emb is not None:
                rope_module = global_rotary_emb
            if rope_module is not None:
                controller.rotary_emb_module = rope_module
            
            # Attach Controller & Monkey Patch
            layer.self_attn.livekv_controller = controller
            self.controllers.append(controller)
            layer.self_attn.forward = types.MethodType(_custom_attention_forward, layer.self_attn)
            
    def _chunk_input(self, input_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        seq_len = input_ids.size(1)
        chunks = []
        for i in range(0, seq_len, chunk_size):
            chunks.append(input_ids[:, i : i + chunk_size])
        return chunks

    def _prefill(self, input_ids: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        [Refactor] 獨立的 Prefill 階段。
        負責：切分 Chunk -> 重置 Controller -> 迴圈計算 -> 產出最後的 Logits。
        """
        chunks = self._chunk_input(input_ids, self.config.chunk_size)
        
        # 重置所有 Controller 狀態
        for controller in self.controllers:
            controller.reset_cache()
            
        current_pos = 0
        last_chunk_logits = None 
        
        # Prefill Loop
        for i, chunk in enumerate(chunks):
            chunk_len = chunk.size(1)
            for controller in self.controllers:
                controller.set_chunk_idx(i)

            position_ids = torch.arange(current_pos, current_pos + chunk_len, device=self.device).unsqueeze(0)
            
            # Forward pass (use_cache=False 因為我們自己管理 KV)
            outputs = self.model(input_ids=chunk, position_ids=position_ids, use_cache=False)
            
            current_pos += chunk_len
            if i == len(chunks) - 1:
                last_chunk_logits = outputs.logits
                
        return current_pos, last_chunk_logits

    def _decode(self, start_logits: torch.Tensor, current_pos: int, max_new_tokens: int, temperature: float) -> List[int]:
        """
        [Refactor] 獨立的 Decoding 階段。
        負責：Sample 第一個 token -> 進入 Decoding 模式 -> 逐字生成。
        """
        # 1. Sample First Token from Prefill result
        next_token_logits = start_logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
        generated_ids = [next_token.item()]
        
        # 2. 通知所有 Controller 切換到 Decoding 模式 (例如保護 Sink Token)
        for controller in self.controllers:
            controller.set_decoding_mode()

        # 3. Decoding Loop
        for _ in range(max_new_tokens - 1):
            position_ids = torch.tensor([[current_pos]], device=self.device)
            outputs = self.model(input_ids=next_token, position_ids=position_ids, use_cache=False)
            
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
                
        return generated_ids

    def generate(self, input_ids: torch.Tensor = None, prompt: str = None, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        主生成入口。
        支援 input_ids (Tensor) 或 prompt (str)。
        """
        # === 1. 輸入標準化 (Input Normalization) ===
        if input_ids is None:
            if prompt is None:
                raise ValueError("Must provide either 'input_ids' or 'prompt'")
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        input_ids = input_ids.to(self.device)

        # === 2. 執行生成流程 (Prefill -> Decode) ===
        with torch.inference_mode():
            # Phase A: Prefill
            current_pos, last_logits = self._prefill(input_ids)
            
            # Phase B: Decode
            generated_ids = self._decode(last_logits, current_pos, max_new_tokens, temperature)
            
        # === 3. 輸出處理 ===
        generated_tensor = torch.tensor(generated_ids, device=self.device)
        output_text = self.tokenizer.decode(generated_tensor, skip_special_tokens=True)
        
        return output_text