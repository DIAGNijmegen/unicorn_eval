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
from bert_score import BERTScorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm
from pathlib import Path
from transformers import logging

logging.set_verbosity_error()


def compute_cider_score(references, predictions):
    """
    Compute CIDEr score for generated captions.

    Args:
        references (list of str): List of reference texts.
        predictions (list of str): List of predicted texts.

    Returns:
        float: CIDEr score.
    """

    scorer = Cider()

    gts = {
        i: ([refs] if isinstance(refs, str) else refs)
        for i, refs in enumerate(references)
    }
    res = {i: [pred] for i, pred in enumerate(predictions)}

    score, _ = scorer.compute_score(gts, res)
    return score


def compute_bleu_score(reports_true, reports_pred):
    """
    Calculate average machine translation metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOR).

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        dict: Dictionary containing averaged scores for BLEU, ROUGE-L, and METEOR.
    """
    # Initialize scorers
    scorer_b = Bleu(4)

    # Prepare data in the required format
    gts = {
        i: [refs] if isinstance(refs, str) else refs
        for i, refs in enumerate(reports_true)
    }
    res = {i: [pred] for i, pred in enumerate(reports_pred)}

    # Compute BLEU scores
    bleu_scores, _ = scorer_b.compute_score(gts, res)

    return bleu_scores[3]


def compute_rouge_score(reports_true, reports_pred):
    """
    Calculate average machine translation metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOR).

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        dict: Dictionary containing averaged scores for BLEU, ROUGE-L, and METEOR.
    """
    # Initialize scorers
    scorer_r = Rouge()

    # Prepare data in the required format
    gts = {
        i: [refs] if isinstance(refs, str) else refs
        for i, refs in enumerate(reports_true)
    }
    res = {i: [pred] for i, pred in enumerate(reports_pred)}

    # Compute ROUGE-L score
    rouge_score, _ = scorer_r.compute_score(gts, res)

    return rouge_score


def compute_meteor_score(reports_true, reports_pred):
    """
    Calculate average machine translation metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOR).

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        dict: Dictionary containing averaged scores for BLEU, ROUGE-L, and METEOR.
    """
    # Initialize scorers
    scorer_m = Meteor()

    # Prepare data in the required format
    gts = {
        i: [refs] if isinstance(refs, str) else refs
        for i, refs in enumerate(reports_true)
    }
    res = {i: [pred] for i, pred in enumerate(reports_pred)}

    # Compute METEOR score
    meteor_score, _ = scorer_m.compute_score(gts, res)

    return meteor_score


def compute_bert_score(reports_true, reports_pred):
    """
    Compute BERTScore (Precision, Recall, F1) for generated text using DeBERTa.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        Tuple containing lists of precision, recall, and F1 scores.
    """
    p_list, r_list, f1_list = [], [], []

    model_directory = "/opt/app/unicorn_eval/models/dragon-bert-base-mixed-domain"
    # ensure the model directory exists
    assert Path(
        model_directory
    ).exists(), f"Model directory {model_directory} does not exist."

    scorer = BERTScorer(
        model_type=model_directory, num_layers=12, lang="nl", device="cpu"  # local path
    )
    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):
        p, r, f1 = scorer.score([text_true], [text_pred])
        p_list.append(p.cpu().numpy().item())
        r_list.append(r.cpu().numpy().item())
        f1_list.append(f1.cpu().numpy().item())
    f1_list = np.array(f1_list)  #
    p_list = np.array(p_list)
    r_list = np.array(r_list)

    average_f1 = f1_list.mean().item()
    return average_f1


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
