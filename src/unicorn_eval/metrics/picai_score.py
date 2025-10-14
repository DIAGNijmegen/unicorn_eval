import os
from contextlib import redirect_stdout, redirect_stderr

with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
    from picai_eval import evaluate
    from report_guided_annotation import extract_lesion_candidates


def compute_picai_score(gts, preds):
    metrics = evaluate(
        y_det=preds,
        y_true=gts,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
    )
    return metrics.score
