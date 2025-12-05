import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List, Optional

# 引入內部模組
from .modules.layer_controller import TransformerLayerController

logger = logging.getLogger(__name__)

class LiveKVQuantModel:
    """
    LiveKVQuant-P 的模型包裝器。
    
    職責：
    1. 載入 HuggingFace 基礎模型 (Llama-3)。
    2. 初始化並注入 TransformerLayerController 到每一層。
    3. 實作 End-to-End 的 Chunk-wise Prefill + Decoding 流程 (Figure 3-1)。
    """

    def __init__(self, model_id: str, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        logger.info(f"Loading tokenizer and model from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 載入模型 (以 FP16 載入以節省記憶體，後續我們會在 Controller 轉 INT4)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # 確保有 pad_token (Llama-3 有時需要手動設定)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # [核心步驟] 注入 Layer Controllers
        self._inject_controllers()
        
        # 設定為評估模式
        self.model.eval()

    def _inject_controllers(self):
        """
        為模型的每一層 (Decoder Layer) 掛載一個 TransformerLayerController。
        對應論文架構圖 Figure 3-2。
        """
        logger.info("Injecting TransformerLayerControllers into model layers...")
        
        # 假設是 Llama 架構 (model.model.layers)
        # 如果是其他架構 (如 Mistral/OPT) 可能需要調整路徑
        self.layers = self.model.model.layers
        self.controllers = []

        for i, layer in enumerate(self.layers):
            # 初始化 Controller
            controller = TransformerLayerController(self.config, layer_idx=i)
            
            # 將 Controller 掛載到 Layer 上
            # 注意：這裡我們暫時使用 '掛載' 方式。
            # 在 modules/layer_controller.py 實作時，我們需要用 Hook 或 Monkey Patch 
            # 讓 Layer 的 forward pass 真正呼叫到 controller。
            # 這裡我們先建立連結。
            layer.livekv_controller = controller
            self.controllers.append(controller)
            
        logger.info(f"Successfully injected {len(self.controllers)} controllers.")

    def _chunk_input(self, input_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """
        將輸入序列切分為多個 Chunk。
        對應論文 Figure 2-6: The Process of Chunking。
        """
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
        
        logger.info(f"Processing Prompt: {input_ids.size(1)} tokens")

        # 2. Divide: 將 Prompt 切分為 Chunks
        chunks = self._chunk_input(input_ids, self.config.chunk_size)
        logger.info(f"Split into {len(chunks)} chunks (Size: {self.config.chunk_size})")

        # 3. Prefill Phase (One by one)
        # 我們需要手動控制 KV Cache 的累積，不能直接 call model.generate
        past_key_values = None 
        
        # 用於重置 Controller 狀態 (例如清空上一輪的 Cache)
        for controller in self.controllers:
            controller.reset_cache()

        logger.info("Starting Prefill Phase...")
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                # 告訴 Controller 目前是第幾個 Chunk (用於判斷 Warm-up)
                for controller in self.controllers:
                    controller.set_chunk_idx(i)

                # Forward pass (計算 logits 並累積 KV Cache)
                # 注意：這裡的 outputs.past_key_values 會包含我們 Controller 處理過的 cache
                # 具體實現取決於 layer_controller 如何攔截 HuggingFace 的 cache 機制
                outputs = self.model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 更新 past_key_values 給下一個 chunk
                past_key_values = outputs.past_key_values

        logger.info("Prefill Completed. Starting Decoding Phase...")

        # 4. Decoding Phase (Generating & Repeat)
        # 接續最後一個 chunk 的 logits 進行生成
        current_input_ids = input_ids # 保持完整 context 用於 tokenizer decode (可優化)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            # 將模式切換為 Decoding (不在 Warmup 也不在 Chunking)
            # Controller 內部會知道現在是 token-by-token
            for controller in self.controllers:
                controller.set_decoding_mode() 

            # 單步生成
            outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            
            # 取樣 (Simple Greedy or Temperature)
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            generated_ids.append(next_token.item())
            
            # 停止條件 (EOS)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # 5. Decode Output
        full_output_ids = torch.cat([input_ids[0], torch.tensor(generated_ids, device=self.device)])
        output_text = self.tokenizer.decode(full_output_ids, skip_special_tokens=True)
        
        return output_text