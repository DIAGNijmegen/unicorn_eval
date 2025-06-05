# Local Evaluation Pipeline Setup
To debug and test your adaptor solution in our evaluation pipeline locally, follow these steps to configure your environment correctly:

## 1. Organizing the Input Folder
Your input data, the output produced by the algorithm Docker container, should be organized using the following structure:

```
input/
├── case1/
│   └── output/
│       └── patch-or-image-neural-representation.json
├── case2/
│   └── output/
│       └── patch-or-image-neural-representation.json
├── predictions.json
```

You can download `patch-neural-representation.json` or `image-neural-representation.json` from Grand Challenge after executing your algorithm on a leaderboard. Downloading these files will help you understand its structure. 

This repository includes an example of the `predictions.json` file, which the platform automatically generates. It maps each platform-generated primary key to the corresponding local case ID. For local debugging, you'll need to modify this file so that it includes all `case IDs` from `mapping.csv`, with each `pk` corresponding to a local folder name.

## 2. Ground Truth Setup
The evaluation also requires ground truth data, structured in the following way:

```
/opt/ml/input/data/ground_truth/
└── Task01_classifying_he_prostate_biopsies_into_isup_scores
    ├── mapping.csv
    └── case1/
        └── isup-grade.json
```
- `mapping.csv` maps case IDs to relevant task metadata (such as modality, domain, and task type)
- `isup-grade.json` contains ground truth annotations for Task01. The format may differ depending on the specific task. However, the evaluation container will handle reading this and doing the metric computation. 

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