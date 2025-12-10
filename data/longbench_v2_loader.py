import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class LongBenchV2Loader:
    """
    [New] 專門負責 LongBench v2 數據集。
    特性:
    1. Dataset ID: 'zai-org/LongBench-v2'
    2. 結構: 單選題 (A/B/C/D)，無 'answers' 列表，只有 'answer' 欄位。
    3. 長度: 8k ~ 2M，適合壓力測試。
    """

    def __init__(self, task_name: str = "all", split: str = "train"):
        """
        Args:
            task_name: 若指定 (如 "Single-Document QA")，則只載入該類別；"all" 則載入全部。
        """
        self.task_name = task_name
        logger.info(f"Loading LongBench v2 dataset: {task_name} (split={split})...")
        
        try:
            # v2 只有一個 'default' config，數據都在 'train' split
            self.dataset = load_dataset("zai-org/LongBench-v2", "default", split=split)
        except Exception as e:
            logger.error(f"Failed to load LongBench v2: {e}")
            raise e

        # 過濾特定任務類別 (Domain)
        if task_name.lower() != "all":
            original_len = len(self.dataset)
            # v2 使用 'domain' 欄位來分類
            self.dataset = self.dataset.filter(lambda x: task_name.lower() in x['domain'].lower())
            logger.info(f"Filtered by domain '{task_name}': {original_len} -> {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def _format_prompt(self, context: str, question: str, choices: dict) -> str:
        """
        將 v2 的選擇題格式化為 Llama-3 偏好的 Prompt。
        """
        # 格式參考 LongBench v2 官方評測
        prompt = (
            f"Read the following context and answer the question by choosing the correct option (A, B, C, or D).\n\n"
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