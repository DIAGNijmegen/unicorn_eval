#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import re
from bert_score import BERTScorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pathlib import Path
from transformers import logging

logging.set_verbosity_error()

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")

def sanitize_text(s: str) -> str:
    """Sanitize a string to remove special characters."""
    if s is None:
        return ""
    if not isinstance(s, str):
        return ""
    s = s.replace("|||", " ")
    s = _CTRL.sub(" ", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _WS.sub(" ", s).strip()
    return s

def sanitize_list(lst):
    """Sanitize a list of strings, allowing for nested lists (per-image multi-refs)."""
    if isinstance(lst, str):
        return [sanitize_text(lst)]
    return [sanitize_text(x) for x in lst]

def prepare_pycoco_inputs(reports_true, reports_pred):
    """
    Returns (gts, res) dictionaries as expected by pycocoevalcap:
    gts: {idx: [ref1, ref2, ...]}, res: {idx: [hyp]}
    Both sanitized.
    """
    gts = {i: sanitize_list(refs) for i, refs in enumerate(reports_true)}
    res = {i: [sanitize_text(pred)] for i, pred in enumerate(reports_pred)}
    return gts, res
    
def compute_cider_score(reports_true, reports_pred):
    """
    Compute CIDEr score for generated captions.

    Args:
        references (list of str): List of reference texts.
        predictions (list of str): List of predicted texts.

    Returns:
        float: CIDEr score.
    """

    scorer = Cider()
    gts, res = prepare_pycoco_inputs(reports_true, reports_pred)
    score, _ = scorer.compute_score(gts, res)
    return score

def compute_bleu_score(reports_true, reports_pred):
    """
    Compute the average BLEU score between reference and predicted reports.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        float: The average BLEU score across all report pairs.
    """
    scorer = Bleu(4)
    gts, res = prepare_pycoco_inputs(reports_true, reports_pred)
    bleu_scores, _ = scorer.compute_score(gts, res)
    return bleu_scores[3]  # BLEU-4

def compute_rouge_score(reports_true, reports_pred):
    """
    Compute the average ROUGE-L score between reference and predicted reports.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        float: The average ROUGE-L score across all report pairs.
    """
    scorer = Rouge()
    gts, res = prepare_pycoco_inputs(reports_true, reports_pred)
    rouge_l, _ = scorer.compute_score(gts, res)
    return rouge_l

def compute_meteor_score(reports_true, reports_pred):
    """
    Compute the average METEOR score between reference and predicted reports.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        float: The average METEOR score across all report pairs.
    """
    scorer = Meteor()
    gts, res = prepare_pycoco_inputs(reports_true, reports_pred)
    meteor, _ = scorer.compute_score(gts, res)
    return meteor


def compute_bert_score(reports_true, reports_pred):
    """
    Compute BERTScore (Precision, Recall, F1) for generated text using a local DeBERTa model.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        float: F1 score averaged across items.
    """
    # Sanitize text
    refs = [sanitize_text(r) for r in reports_true]
    cands = [sanitize_text(p) for p in reports_pred]

    model_directory = "/opt/app/unicorn_eval/models/dragon-bert-base-mixed-domain"
    assert Path(model_directory).exists(), f"Model directory {model_directory} does not exist."

    # Load scorer once
    scorer = BERTScorer(
        model_type=model_directory,
        num_layers=12,
        lang="nl",
        device="cpu"
    )

    # Vectorized scoring 
    _, _, F1 = scorer.score(cands, refs)

    # Convert to numpy arrays
    f1_list = F1.cpu().numpy()

    return float(f1_list.mean())


def compute_average_language_metric(reports_true, reports_pred):
    """
    Compute average language evaluation metrics for generated text.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of predicted texts.

    Returns:
        dict: Dictionary containing averaged scores for CIDEr, BLEU, ROUGE-L, METEOR, and BERTScore F1.
    """
    
    metrics, normalized_metrics = {}, {}

    metric_info = {
        "CIDEr": {"fn": compute_cider_score, "range": (0, 10)},
        "BLEU-4": {"fn": compute_bleu_score, "range": (0, 1)},
        "ROUGE-L": {"fn": compute_rouge_score, "range": (0, 1)},
        "METEOR": {"fn": compute_meteor_score, "range": (0, 1)},
        "BERTScore_F1": {"fn": compute_bert_score, "range": (0, 1)},
    }
    for metric_name, metric_details in metric_info.items():
        metric_fn = metric_details["fn"]
        metric_value = metric_fn(reports_true, reports_pred)
        min_value, max_value = metric_details["range"]
        normalized_value = (metric_value - min_value) / (max_value - min_value)
        metrics[metric_name] = metric_value
        normalized_metrics[metric_name] = normalized_value

    # compute average of normalized metrics
    average_normalized_metric = np.mean(list(normalized_metrics.values()))
    metrics["average_language_metric"] = average_normalized_metric
    return metrics
