# data/constants.py

# === 任務定義 ===
V1_TASK_GROUPS = {
    "single-doc": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi-doc": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "synthetic": ["passage_retrieval_en", "passage_count"],
    "code": ["lcc", "repobench-p"],
    "all": [
        "narrativeqa", "qasper", "multifieldqa_en", 
        "hotpotqa", "2wikimqa", "musique", "dureader", 
        "gov_report", "qmsum", "multi_news", "vcsum",
        "trec", "triviaqa", "samsum", "lsht",
        "passage_retrieval_en", "passage_count",
        "lcc", "repobench-p"
    ]
}

V2_DOMAIN_MAP = {
    "single-doc": "single-document",
    "multi-doc": "multi-document",
    "summarization": "summarization",
    "long-context": "long in-context",
    "dialogue": "long dialogue",
    "code": "code repository",
    "all": "all"
}

def get_task_list(version, task_type):
    """
    根據版本和輸入的 task_type 字串回傳任務列表。
    這段邏輯原本在兩個 script 裡都有，現在統一在這裡管理。
    """
    task_type_lower = task_type.lower()
    
    if version == "v1":
        # 如果是群組名稱 (如 single-doc)，回傳列表
        if task_type_lower in V1_TASK_GROUPS:
            return V1_TASK_GROUPS[task_type_lower]
        # 否則假設是單一任務名稱
        return [task_type]
        
    elif version == "v2":
        # V2 的邏輯是用 domain map
        domain = V2_DOMAIN_MAP.get(task_type_lower, task_type)
        return [domain]
    
    return [task_type]