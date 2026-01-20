import logging
import os
from datasets import load_dataset
from .constants import LONGBENCH_PROMPTS  # [新增] 引入 Prompt 定義

logger = logging.getLogger(__name__)

class LongBenchLoader:
    """
    負責載入 LongBench 數據集。
    優先嘗試讀取本地 JSONL 檔案，若無則嘗試從 HuggingFace Hub 下載。
    """

    SUPPORTED_TASKS = [
        "narrativeqa", "qasper", "multifieldqa_en", 
        "hotpotqa", "2wikimqa", "musique", "dureader", 
        "gov_report", "qmsum", "multi_news", "vcsum",
        "trec", "triviaqa", "samsum", "lsht",
        "passage_retrieval_en", "passage_count",
        "lcc", "repobench-p"
    ]

    def __init__(self, task_name: str, split: str = "test", local_data_dir: str = "data/longbench_v1/"):
        if task_name not in self.SUPPORTED_TASKS:
            logger.warning(f"Task '{task_name}' is not in the standard LongBench list.")
        
        self.task_name = task_name
        local_file_path = os.path.join(local_data_dir, f"{task_name}.jsonl")
        
        if os.path.exists(local_file_path):
            try:
                dataset = load_dataset("json", data_files=local_file_path, split="train")
                self.dataset = dataset
            except Exception as e:
                logger.error(f"Failed to load local file {local_file_path}: {e}")
                raise e
        else:
            logger.warning(f"Local file not found at {local_file_path}. Attempting to load from HuggingFace Hub...")
            dataset_path = "THUDM/LongBench"
            try:
                self.dataset = load_dataset(dataset_path, task_name, split=split, trust_remote_code=True)
            except Exception as e:
                logger.error(f"Failed to load from HF Hub: {e}")
                raise e

    def __len__(self):
        return len(self.dataset)

    def _format_prompt(self, context: str, input_text: str) -> str:
        """
        [修改] 根據 Task Name 自動選擇正確的 Prompt Template。
        """
        # 1. 優先使用 constants.py 定義的官方 Template
        if self.task_name in LONGBENCH_PROMPTS:
            template = LONGBENCH_PROMPTS[self.task_name]
            return template.format(context=context, input=input_text)
            
        # 2. 預設 Fallback 格式
        return f"Context:\n{context}\n\nQuestion:\n{input_text}\n\nAnswer:"

    def get_sample(self, index: int):
        data = self.dataset[index]
        context = data['context']
        input_text = data['input']
        answers = data['answers'] 
        length = data['length']

        # 這裡回傳的 prompt 已經是最終可用的格式了
        formatted_prompt = self._format_prompt(context, input_text)
        
        return {
            "prompt": formatted_prompt,
            "answers": answers,
            "context_length": length,
            # 依然保留 raw data 以備不時之需 (例如 Debug)
            "raw_context": context,
            "raw_input": input_text
        }

    def get_dataset(self):
        return self.dataset