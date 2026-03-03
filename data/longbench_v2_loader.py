import logging
import os
import json
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

class LongBenchV2Loader:
    def __init__(self, task_name: str = "all", split: str = "train", data_dir: str = "data/longbench_v2/"):
        """
        Args:
            task_name: Domain 名稱 (例如 "Single-Document QA")
            data_dir: divide.py 儲存檔案的資料夾
        """
        self.task_name = task_name
        
        # 1. 建立對應的本地檔名 (依照 divide.py 的邏輯：空格變底線，橫線變底線)
        # 例如 "Single-Document QA" -> "Single_Document_QA.json"
        sanitized_name = task_name.replace(" ", "_").replace("-", "_")
        local_path = os.path.join(data_dir, f"{sanitized_name}.json")

        # 2. 優先讀取本地檔案
        if os.path.exists(local_path) and task_name.lower() != "all":
            logger.info(f"Loading LongBench v2 from local file: {local_path}")
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # 將 List 轉換為 HuggingFace Dataset 格式，維持後續 API 一致性
                self.dataset = Dataset.from_list(raw_data)
        else:
            # 如果本地沒檔案或指定 "all"，則走原始載入邏輯
            logger.info(f"Loading LongBench v2 from HuggingFace (Domain: {task_name})...")
            try:
                self.dataset = load_dataset('THUDM/LongBench-v2', split=split)
                if task_name.lower() != "all":
                    self.dataset = self.dataset.filter(
                        lambda x: task_name.lower() in x['domain'].lower()
                    )
            except Exception as e:
                logger.error(f"Failed to load LongBench v2: {e}")
                raise e

        logger.info(f"Successfully loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def _format_prompt(self, context: str, question: str, choices: dict) -> str:
        """
        將 v2 的選擇題格式化為 Llama-3 偏好的 Prompt。
        """
        # 格式參考 LongBench v2 官方評測
        prompt = (
            f"Read the following context and answer the question by choosing the correct option (A, B, C, or D).\n\n"
            f"Please output ONLY the single letter (A, B, C, or D) of the correct option and nothing else.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n"
            f"A. {choices.get('choice_A', '')}\n"
            f"B. {choices.get('choice_B', '')}\n"
            f"C. {choices.get('choice_C', '')}\n"
            f"D. {choices.get('choice_D', '')}\n\n"
            f"Answer:"
        )
        return prompt

    def get_sample(self, index: int):
        data = self.dataset[index]
        
        # 提取選項
        choices = {
            'choice_A': data.get('choice_A', ''),
            'choice_B': data.get('choice_B', ''),
            'choice_C': data.get('choice_C', ''),
            'choice_D': data.get('choice_D', '')
        }
        
        formatted_prompt = self._format_prompt(
            data['context'], 
            data['question'], 
            choices
        )

        return {
            "prompt": formatted_prompt,
            # 為了相容 metrics.py，我們把單一答案包成 list
            "answers": [data['answer']], 
            "context_length": data['length'], # v2 直接提供了 length 欄位
            "domain": data['domain'],
            "difficulty": data.get('difficulty', 'unknown')
        }

    def get_dataset(self):
        return self.dataset