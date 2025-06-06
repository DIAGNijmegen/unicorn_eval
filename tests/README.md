# Local Evaluation Pipeline Setup
To debug and test your adaptor solution in our evaluation pipeline locally, follow these steps to configure your environment correctly:

## 1. Organizing the Input Folder
Your input data, the output produced by the algorithm Docker container, should be organized using the following structure:

```
input/
├── pk-value/
│   └── output/
│       └── patch-or-image-neural-representation.json
├── pk-value/
│   └── output/
│       └── patch-or-image-neural-representation.json
├── predictions.json
├── adaptor-task-file.json       // e.g. adaptor-pathology-classification.json
```

You can download `patch-neural-representation.json` or `image-neural-representation.json` from Grand Challenge after executing your algorithm on a leaderboard. Downloading these files will help you understand its structure. 

This repository includes an example of a simplified `predictions.json` file, which the platform automatically generates. It maps each platform-generated primary key to the corresponding local case ID. For local debugging, you'll need to modify this file so that it includes all `case IDs` from `mapping.csv`, with each `pk` corresponding to a local folder name.

The simplified `predictions.json` file follows this structure: 
```
[
  {
    "pk": <pk-value>,       // pk-value is same as case_id
    "inputs": [
      {
        "image": {
          "name": <case_id>       // matches case IDs in mapping.csv
        },
        "interface": {
          "slug": <input_slug>       // retrieve the input slug for each task from INPUT_SLUGS_DICT (defined in evaluate.py)
        }
      }
    ],
    "outputs": [
      {
        "interface": {
          "slug": <neural-representation-type>,       // either "image-neural-representation" or "patch-neural-representation"
          "relative_path": <json-file-name>           // either "image-neural-representation.json" or "patch-neural-representation.json"
        }
      }
    ]
  }
]
```

The `adaptor-task-file.json` file is a task-specific file that indicates which adaptor method to use. Its filename (e.g., adaptor-pathology-classification.json) can be retrieved from ADAPTOR_SLUGS_DICT (defined in `evaluate.py`). The content of this file is the adaptor name, which can be selected when submitting to a task on Grand Challenge—such as "1-nn", "linear-probing", "mlp", ... .

## 2. Ground Truth Setup
The evaluation also requires ground truth data, structured in the following way:

``` 
/opt/ml/input/data/ground_truth/
└── <task_name>       // e.g. Task01_classifying_he_prostate_biopsies_into_isup_scores
    ├── mapping.csv
    └── pk-value
        └── label       // e.g. isup-grade.json
├── mapping.csv
```

The `mapping.csv` maps case IDs to relevant task metadata, such as task_name and domain, and specifies whether the case is used for training (“shot”) or testing (“case”).

```
case_id   | task_name | task_type                           | domain            | modality | split
<case_id> | <task_name> | classification/detection/segmentation | pathology/CT/MR | vision   | shot/case
``` 

**Important:** The `name` field in `predictions.json` must exactly match the case ID used in `mapping.csv`. If they don't align, the evaluation container will be unable to associate predictions with the correct ground truth data.

## 3. Running the Evaluation Locally
To test your setup, either:
- Download your `patch-neural-representations.json` from Grand Challenge, or
- Run your algorithm locally on few-shot data (available on Zenodo) and generate dummy files. See [our publicly available baseline setup](https://github.com/DIAGNijmegen/unicorn_baseline/blob/main/setup-docker.md) with an example of how to run the baseline on for example the public few-shot data.
Make sure that:
- The folder structure is setup as described above. The Docker will automatically look for the files in the predefined folders.
- Case IDs are consistent across `predictions.json`, `mapping.csv`, and the input directory.

With these structures in place, you can validate that the evaluation pipeline works from start to finish and start developing your own adaptor solution.

## 4. Example Data
**You can use publicly released [few-shot example data on Zenodo](10.5281/zenodo.14832502) as a reference**. These datasets include sample inputs and labels for each task, helping you build and test your local environment. 

We recommend generating a custom `mapping.csv` based on the provided template and creating your own data split. For example, you could divide the few-shot examples into 30% "shot" and 70% "case" for experimentation.

---
**If you're using files downloaded from Grand Challenge**, we suggest following the same approach, create a custom `mapping.csv` and define a split as needed. However, since ground truth annotations are not provided through Grand Challenge downloads, you’ll need to simulate the ground truth setup. To do this, we recommend referring to the Zenodo examples to replicate the structure and format of the ground truth files.