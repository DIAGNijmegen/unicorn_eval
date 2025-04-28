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
    - A short `README.md` that covers:
      - A clear description of your method
      - A list of tasks, or task types your method is designed for
    - A **unique name** (we’ll include your **team name** in the adaptor name to ensure you receive credit)
    - Any additional dependencies in a `requirements.txt` (details on adding new requirements below)

✅ Once accepted, your adaptor becomes selectable at submission time — and your team gets full recognition when it’s used!

> 💡 Keep in mind: we **prioritize originality**. If your adaptor is too similar to an existing one, it may not be accepted — so submit early and make it your own!


## 📚 Guidelines for Contributing a New Adaptor

Developed your own adaptor and want to get it into the official repo? Here’s what you need to consider:

**Implementation requirements**
- Your adaptor method must be implemented as a standalone function, following the baseline template [`base.py`](src/unicorn_eval/adaptors/base.py)
- It must complete within the allowed time limit of 1h
- Adaptors must be designed to run on CPU
- Submissions will be evaluated for correctness, efficiency, and compliance with the [challenge policies](https://unicorn.grand-challenge.org/requirements-and-guidelines/)
- 🚨 Important: Pre-trained adaptors are not allowed! Be original — you can use the few-shots, for example, for fitting or training your adaptor, but don’t rely on pre-trained solutions

**Environment and dependencies**
- Each method must be able to run in the [provided isolated environment](https://github.com/DIAGNijmegen/unicorn_eval/blob/improve-code-readability/Dockerfile)
- Additional dependencies can be requested but:
  - Approval of new dependencies is not guaranteed, dependencies will be evaluated based on compatibility with other packages
  - Organizers reserve the right to modify the list of dependencies over time, though we aim to maintain compatibility with existing adaptors
  - When specifying dependencies, use the least restrictive version (e.g., package>=1.0.0) to ensure flexibility

> 💬 Teams are encouraged to share ideas and discuss approaches on the [Grand Challenge forum](https://grand-challenge.org/forums/forum/unicorn-740/). Support and Q&A will also be available through the forum.

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