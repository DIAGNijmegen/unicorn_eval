from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate


def compute_picai_score(gts, preds):
    metrics = evaluate(
        y_det=preds,
        y_true=gts,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
    )
    return metrics.auroc
