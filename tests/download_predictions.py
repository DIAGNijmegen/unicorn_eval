import json
import os
from pathlib import Path

import gcapi
from tqdm import tqdm


# paths
def download_predictions(
    output_dir: Path = Path("predictions"),
    predictions_path: Path | None = None,
):
    if predictions_path is None:
        predictions_path = output_dir / "predictions.json"

    # setup
    with open(predictions_path) as fp:
        predictions = json.load(fp)

    token = os.environ["GC_TOKEN"]
    client = gcapi.Client(token=token)

    # retrieve predictions from the submission
    for pred in tqdm(predictions):
        # create output folder
        out_dir: Path = output_dir / pred["pk"] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # download results
        outputs = pred["outputs"]
        for output in outputs:
            # download the predictions
            dest_path = out_dir / f'{output["interface"]["slug"]}.json'
            if dest_path.exists():
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'w') as fp:
                predictions = client(url=output["file"], follow_redirects=True)
                json.dump(predictions, fp)


if __name__ == "__main__":
    download_predictions(
        output_dir=Path("/Users/joeranbosma/repos/unicorn_eval/tests/vision/input-task10-val"),
    )
