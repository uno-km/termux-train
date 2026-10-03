"""
termux_train.rl.reward
======================
Rule-Based Verifiers and Reward Functions for On-Device Reinforcement Learning.
Enforces structural formatting (XML <think> tags like DeepSeek-R1), exact-match accuracy,
and length penalty without requiring a separate critic model.
"""

from __future__ import annotations

import re
from typing import List, Optional, Union, Dict, Any


def xml_format_reward(
    text: str,
    tags: Optional[List[str]] = None,
    allow_nesting: bool = False
) -> float:
    """
    Rewards structured reasoning output with XML tags (e.g. <think>...</think><answer>...</answer>).
    Returns 1.0 if strictly compliant, partial credit if partially compliant, or 0.0 if missing.
    """
    if tags is None:
        tags = ["think", "answer"]

    if not isinstance(text, str) or not text.strip():
        return 0.0

    score = 0.0
    weight_per_tag = 1.0 / len(tags)

    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        has_open = open_tag in text
        has_close = close_tag in text
        
        if has_open and has_close:
            # Check ordering: open tag must precede close tag
            open_pos = text.find(open_tag)
            close_pos = text.rfind(close_tag)
            if open_pos < close_pos:
                content = text[open_pos + len(open_tag):close_pos].strip()
                if content:
                    score += weight_per_tag
                else:
                    score += weight_per_tag * 0.5  # Empty tag penalized

    return min(max(score, 0.0), 1.0)


def extract_answer_content(text: str, tag: str = "answer") -> str:
    """Extracts content inside <tag>...</tag> or returns cleaned raw text."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def accuracy_reward(
    prediction: str,
    ground_truth: str,
    strict: bool = False
) -> float:
    """
    Evaluates prediction correctness against ground truth.
    Supports exact string match or numeric token extraction.
    """
    pred_ans = extract_answer_content(prediction)
    gt_ans = ground_truth.strip()

    if strict:
        return 1.0 if pred_ans == gt_ans else 0.0

    # Case-insensitive whitespace-normalized match
    norm_pred = " ".join(pred_ans.lower().split())
    norm_gt = " ".join(gt_ans.lower().split())
    if norm_pred == norm_gt:
        return 1.0

    # Numeric match if ground truth is a number
    try:
        gt_num = float(norm_gt.replace(",", ""))
        # Search for number in prediction
        pred_numbers = re.findall(r"[-+]?\d*\.?\d+", norm_pred)
        if pred_numbers:
            last_num = float(pred_numbers[-1])
            if abs(last_num - gt_num) < 1e-5:
                return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


def length_penalty_reward(
    text: str,
    max_tokens: int = 512,
    target_tokens: int = 256,
    tolerance: float = 0.2
) -> float:
    """
    Penalizes overly repetitive or excessively verbose text.
    Returns value in range [-1.0, 0.0].
    """
    # Approximate tokens by whitespace count
    token_count = len(text.split())
    if token_count > max_tokens:
        excess = token_count - max_tokens
        penalty = min(excess / max_tokens, 1.0)
        return -penalty
    return 0.0
