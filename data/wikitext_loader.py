import logging
import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class WikitextLoader:
    """
    負責載入 Wikitext-2 數據集，用於計算 Perplexity (PPL)。
    對應論文 Method Summary 中的 Metrics: Perplexity = exp(cross_entropy).
    """

    def __init__(self, dataset_name: str = "wikitext-2-raw-v1", split: str = "test"):
        logger.info(f"Loading Wikitext dataset: {dataset_name} (split={split})...")
        try:
            self.dataset = load_dataset("wikitext", dataset_name, split=split)
        except Exception as e:
            logger.error("Failed to load Wikitext. Check internet or dataset name.")
            raise e

    def get_text(self) -> str:
        """
        取得合併後的純文本 (Raw Text)。
        Wikitext 原始數據是按行分開的，計算 PPL 時通常需要合併為長字串。
        """
        logger.info("Merging Wikitext lines into a single stream...")
        # 過濾掉空行，並用換行符號連接
        text_stream = "\n\n".join([x for x in self.dataset["text"] if x.strip()])
        return text_stream

    def get_tokenized_stream(self, tokenizer):
        """
        將文本轉換為 Token ID Stream，用於 PPL 計算。
        
        Args:
            tokenizer: HuggingFace tokenizer
            
        Returns:
            torch.Tensor: 形狀為 [1, seq_len] 的超長 Tensor
        """
        text = self.get_text()
        logger.info(f"Tokenizing text stream (Total chars: {len(text)})...")
        
        # Tokenize 整個文本 (不截斷，因為我們要測量完整的 PPL)
        encodings = tokenizer(text, return_tensors="pt")
        
        logger.info(f"Tokenized sequence length: {encodings.input_ids.size(1)}")
        return encodings.input_ids

    @staticmethod
    def chunk_dataset(input_ids, block_size):
        """
        輔助函式：將超長 Tensor 切割成固定大小的 chunks (用於 batch processing)
        雖然 PPL 通常用 Sliding Window，但如果記憶體不足，可以切塊計算。
        """
        total_length = input_ids.size(1)
        # 捨棄最後不足 block_size 的部分
        total_length = (total_length // block_size) * block_size
        
        # Reshape
        input_ids = input_ids[:, :total_length]
        input_ids = input_ids.view(-1, block_size)
        
        return input_ids