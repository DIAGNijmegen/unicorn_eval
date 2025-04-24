# 🧪 UNICORN Evaluation Toolkit

Welcome to the official evaluation repository for the [UNICORN Challenge](https://unicorn.grand-challenge.org/) — a benchmark for foundation models in pathology, radiology, and medical language processing. This repository provides the official evaluation code and a library of **adaptors** used to turn frozen features into predictions in **vision tasks**.

[![PyPI version](https://img.shields.io/pypi/v/unicorn-eval)](https://pypi.org/project/unicorn-eval/)

## 🚀 Challenge Overview

The UNICORN Challenge evaluates how well foundation models generalize across multiple modalities with minimal task-specific supervision:

- 🧠 **Language** and **Vision-Language** tasks: your model directly outputs predictions.
- 👁️ **Vision** tasks: your model outputs features. These are then converted to predictions using **adaptors** — lightweight models like k-NN, linear classifiers, or shallow MLPs.

We provide a few built-in adaptors, but you're highly encouraged to propose your own!<br>
We maintain the full list of adaptors available on the [Supported Adaptors](src/unicorn_eval/adaptors/README.md) page.


## 🧩 Contributing a Custom Adaptor

Have a better idea for how to turn features into predictions?

You’re welcome to contribute a custom adaptor! Here's how:

1. Add your adaptor to `src/unicorn_eval/adaptors/`.
2. Inherit from one of the base adaptor classes in [`base.py`](src/unicorn_eval/adaptors/base.py).
3. Open a pull request with:
    - Your adaptor code
    - A short description
    - A **unique name** (we’ll include your **team name** in the adaptor name to ensure you receive credit).

✅ Once accepted, your adaptor becomes selectable at submission time — and your team gets full recognition when it’s used!

> 💡 Keep in mind: we **prioritize originality**. If your adaptor is too similar to an existing one, it may not be accepted — so submit early and make it your own!

## 📦 Adaptors vs. Algorithms: What's the Difference?

In **vision tasks**, submissions consist of:
- A **feature extractor** (your algorithm)
- An **adaptor** (used to turn features into predictions)

You can experiment with different adaptors **on top of the same algorithm** without using up your submission slots.<br>
Want to try a different adaptor? Send us a request by email, we’ll run the new adaptor strategy for you on top of the existing features. Requests should be submitted via email using the provided template (to be shared soon).

In **language** and **vision-language** tasks, the algorithm outputs predictions directly, so no adaptor is needed.

## Summary

| **Modality**         | **What You Submit**                        | **Are Adaptors Used?** | **Submission Limit Applies To** |
|-----------------------|--------------------------------------------|-------------------------|-----------------------------------|
| **Vision**            | Algorithm (feature extractor) + Adaptor   | ✅ Yes                  | Algorithm only                   |
| **Language**          | Algorithm (predictive)                    | ❌ No                   | Algorithm                        |
| **Vision-Language**   | Algorithm (predictive)                    | ❌ No                   | Algorithm                        |