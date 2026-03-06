import torch
import string
import re
from collections import Counter
from typing import List, Union
from rouge import Rouge

_rouge_scorer = Rouge()

def calculate_perplexity(loss: torch.Tensor) -> float:
    """
    計算 Perplexity (PPL)。
    """
    return torch.exp(loss).item()

def normalize_answer(s: str) -> str:
    """
    標準化答案字串：轉小寫、去除標點符號、去除多餘空白。
    [新增] 強力去除 Llama-3 常用的廢話前綴，大幅提升 F1 Score。
    """
    s = s.lower()
    
    # === [關鍵修正] 去除常見的 Chatty Prefix ===
    # 這些是 Llama-3-Instruct 最愛講的廢話，必須砍掉才能正確算分
    prefixes = [
        "the answer is", 
        "the answer to the question is",
        "based on the context", 
        "based on the text", 
        "according to the passage",
        "according to the text",
        "sure, here is the answer",
        "here is the answer",
        "answer:",
    ]
    
    for prefix in prefixes:
        if s.strip().startswith(prefix):
            s = s.replace(prefix, "", 1)
            
    # 去除括號內的引用說明 (例如: "Harry Potter (Page 5)")
    s = re.sub(r'\([^)]*\)', '', s)
    # =======================================

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))

def calculate_f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算 F1 Score (Bag of Words overlap)。
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
    計算 Accuracy。
    與 LongBench 官方一致：檢查 ground_truth 是否包含在 prediction 中。
    適用於分類任務（TREC）和合成任務（passage_count, passage_retrieval_en），
    因為模型通常會在正確答案前後加入多餘的文字。
    """
    if isinstance(ground_truth, list):
        scores = [calculate_accuracy(prediction, gt) for gt in ground_truth]
        return max(scores) if scores else 0.0

    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # 先嘗試 exact match，再嘗試 containment（與 LongBench 官方一致）
    if pred_norm == gt_norm:
        return 1.0
    if gt_norm in pred_norm:
        return 1.0
    return 0.0


def calculate_rouge_l(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算 Rouge-L，使用 py-rouge 套件，與 LongBench 官方評估一致。
    py-rouge 在 sentence level 計算 LCS 再匯總，比全文 LCS 更合理。
    """
    if isinstance(ground_truth, list):
        scores = [calculate_rouge_l(prediction, gt) for gt in ground_truth]
        return max(scores) if scores else 0.0

    try:
        scores = _rouge_scorer.get_scores(prediction, ground_truth)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0


def calculate_edit_similarity(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算 Code Similarity，與 FastKV 官方一致：使用 fuzz.ratio (SequenceMatcher)。
    Post-processing（過濾 backtick/comment 行）也在此處理，與 FastKV code_sim_score 一致。
    用於 Code Completion 任務 (lcc, repobench-p)。
    """
    from fuzzywuzzy import fuzz

    if isinstance(ground_truth, list):
        scores = [calculate_edit_similarity(prediction, gt) for gt in ground_truth]
        return max(scores) if scores else 0.0

    # Post-processing: extract first valid code line (Reference: FastKV code_sim_score)
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break

    return fuzz.ratio(prediction, ground_truth) / 100