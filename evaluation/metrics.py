import torch
import string
import re
from collections import Counter
from typing import List, Union

def calculate_perplexity(loss: torch.Tensor) -> float:
    """
    計算 Perplexity (PPL)。
    對應論文 Page 44 公式: Perplexity = exp(cross entropy loss) 
    
    Args:
        loss (torch.Tensor): 由 CrossEntropyLoss 計算出的純量值
    Returns:
        float: PPL 值
    """
    return torch.exp(loss).item()

def normalize_answer(s: str) -> str:
    """
    標準化答案字串：轉小寫、去除標點符號、去除多餘空白。
    這是 LongBench 與 SQuAD 評估的標準前處理。
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算 F1 Score (Bag of Words overlap)。
    對應論文 Page 44 公式: F1 = 2 * (Precision * Recall) / (Precision + Recall) 
    
    Args:
        prediction (str): 模型生成的答案
        ground_truth (str or List[str]): 正確答案 (LongBench 通常有多個可接受答案)
    """
    # 處理多個 Ground Truth 的情況，取最高分
    if isinstance(ground_truth, list):
        scores = [calculate_f1_score(prediction, gt) for gt in ground_truth]
        return max(scores) if scores else 0.0

    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # 如果標準化後為空
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_accuracy(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算 Accuracy (Exact Match)。
    對應論文 Page 44 公式: Acc = Correct / Total 
    這裡計算單筆樣本是否命中 (1.0 or 0.0)，Batch 計算時再取平均。
    """
    if isinstance(ground_truth, list):
        scores = [calculate_accuracy(prediction, gt) for gt in ground_truth]
        return max(scores) if scores else 0.0
        
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0