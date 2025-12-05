import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class LongBenchLoader:
    """
    負責載入 LongBench 數據集，用於評估長文本理解能力 (Accuracy, F1 Score)。
    對應論文 Method Summary 中的 Evaluation Metrics。
    """
    
    # LongBench 支援的任務列表 (參考 LongBench 官方 repo)
    SUPPORTED_TASKS = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", 
        "2wikimqa", "musique", "gov_report", "qmsum", 
        "multi_news", "trec", "triviaqa", "samsum"
    ]

    def __init__(self, task_name: str, split: str = "test"):
        if task_name not in self.SUPPORTED_TASKS:
            logger.warning(f"Task '{task_name}' is not in the standard LongBench list. Attempting to load anyway...")
        
        self.task_name = task_name
        logger.info(f"Loading LongBench dataset: {task_name} (split={split})...")
        
        # 使用 HuggingFace Datasets 載入 THUDM/LongBench
        try:
            self.dataset = load_dataset("THUDM/LongBench", task_name, split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}. Please check your internet connection or task name.")
            raise e

    def __len__(self):
        return len(self.dataset)

    def _format_prompt(self, context: str, input_text: str) -> str:
        """
        將 Context 與 Input 格式化為模型可接受的 Prompt。
        這裡使用 LongBench 官方推薦的標準格式。
        """
        # 這是 LongBench 官方對大多數 QA 任務的默認 template
        # 你可以根據 Llama-3 的需求，在這裡加入 <|begin_of_text|> 等 special tokens
        prompt = f"Context:\n{context}\n\nQuestion:\n{input_text}\n\nAnswer:"
        return prompt

    def get_sample(self, index: int):
        """
        取得單筆測試資料。
        
        Returns:
            prompt (str): 組合好的完整輸入 Prompt (包含 Context)
            answers (list): Ground Truth 答案列表 (用於計算 F1/Exact Match)
            length (int): Context 的原始長度
        """
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