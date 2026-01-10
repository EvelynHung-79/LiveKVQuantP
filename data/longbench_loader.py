import logging
import os
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

class LongBenchLoader:
    """
    負責載入 LongBench 數據集。
    優先嘗試讀取本地 JSONL 檔案，若無則嘗試從 HuggingFace Hub 下載。
    """
    
    # 完整的 LongBench V1 任務列表 (包含中文與代碼任務)
    SUPPORTED_TASKS = [
        # Single-Doc QA
        "narrativeqa", "qasper", "multifieldqa_en", 
        # Multi-Doc QA
        "hotpotqa", "2wikimqa", "musique", "dureader", 
        # Summarization
        "gov_report", "qmsum", "multi_news", "vcsum",
        # Few-Shot Learning
        "trec", "triviaqa", "samsum", "lsht",
        # Synthetic Tasks
        "passage_retrieval_en", "passage_count",
        # Code
        "lcc", "repobench-p"
    ]

    def __init__(self, task_name: str, split: str = "test", local_data_dir: str = "data/longbench_v1/"):
        """
        Args:
            task_name: 任務名稱 (e.g., 'narrativeqa')
            split: 數據集分割 (通常是 'test')
            local_data_dir: 本地資料夾路徑，預設為 'LongBench' (請確保您的 .jsonl 檔放在此資料夾下)
        """
        if task_name not in self.SUPPORTED_TASKS:
            logger.warning(f"Task '{task_name}' is not in the standard LongBench list.")
        
        self.task_name = task_name
        
        # 1. 嘗試構建本地檔案路徑
        # 假設結構為: local_data_dir/task_name.jsonl (例如: LongBench/narrativeqa.jsonl)
        local_file_path = os.path.join(local_data_dir, f"{task_name}.jsonl")
        
        if os.path.exists(local_file_path):
            logger.info(f"Found local dataset file: {local_file_path}")
            try:
                # 使用 json loader 讀取本地檔案
                # split='train' 是因為 json loader 預設只會載入到 train，我們後面再手動對應
                dataset = load_dataset("json", data_files=local_file_path, split="train")
                self.dataset = dataset
            except Exception as e:
                logger.error(f"Failed to load local file {local_file_path}: {e}")
                raise e
        else:
            # 2. 本地找不到，嘗試從 HF Hub 下載 (備用方案)
            logger.warning(f"Local file not found at {local_file_path}. Attempting to load from HuggingFace Hub...")
            dataset_path = "THUDM/LongBench"
            try:
                # 注意: trust_remote_code 可能在某些環境被禁用
                self.dataset = load_dataset(dataset_path, task_name, split=split, trust_remote_code=True)
            except Exception as e:
                logger.error(f"Failed to load from HF Hub: {e}")
                logger.error(f"Please ensure you have downloaded '{task_name}.jsonl' into the '{local_data_dir}' folder.")
                raise e

    def __len__(self):
        return len(self.dataset)

    def _format_prompt(self, context: str, input_text: str) -> str:
        """
        將 Context 與 Input 格式化為模型可接受的 Prompt。
        這裡使用 LongBench 官方推薦的標準格式。
        """
        prompt = f"Context:\n{context}\n\nQuestion:\n{input_text}\n\nAnswer:"
        return prompt

    def get_sample(self, index: int):
        data = self.dataset[index]
        context = data['context']
        input_text = data['input']
        answers = data['answers'] # 這是一個 list，因為可能有多個正確寫法
        length = data['length']

        formatted_prompt = self._format_prompt(context, input_text)
        
        return {
            "prompt": formatted_prompt,
            "answers": answers,
            "context_length": length,
            "raw_context": context,
            "raw_input": input_text
        }

    def get_dataset(self):
        """回傳原始 HF Dataset 物件"""
        return self.dataset