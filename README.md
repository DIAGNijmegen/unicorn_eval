# 🧪 UNICORN Evaluation Toolkit

Welcome to the official evaluation repository for the [UNICORN Challenge](https://unicorn.grand-challenge.org/) — **a benchmark for foundation models in pathology, radiology, and medical language processing**. 
This repository provides:
- The official UNICORN Challenge evaluation code
- A growing library of **adaptors** used to turn frozen features into predictions in **vision tasks**.

[![PyPI version](https://img.shields.io/pypi/v/unicorn-eval?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/unicorn-eval/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/unicorn_eval?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/unicorn_eval)

## 🚀 Challenge Overview

The UNICORN Challenge evaluates how well foundation models generalize across multiple modalities with minimal task-specific supervision:

- 🧠 **Language** and **Vision-Language** tasks: algorithm directly outputs predictions — _no adaptor required_ 
- 👁️ **Vision** tasks: algorithms outputs frozen features, these are passed through **adaptors** — lightweight models - to generate predictions. 

We provide a few built-in adaptors, but you're highly encouraged to propose your own!<br>
We maintain the full list of adaptors available on the [Supported Adaptors](src/unicorn_eval/adaptors/README.md) page.

## 📦 Adaptors vs. Algorithms: What's the Difference?

In **vision tasks**, submissions consist of:
- A **feature extractor** (your algorithm)
- An **adaptor** (used to turn features into predictions)

You can experiment with different adaptors **on top of the same algorithm** without using up your submission slots.<br>
Want to try a different adaptor? Email us using the provided template (see below) — we’ll run it for you on existing features.

In **language** and **vision-language** tasks, the algorithm outputs predictions directly, so no adaptor is needed.

## 🧩 Contributing a Custom Adaptor 

Have a better idea for how to turn features into predictions?

### You’re welcome to contribute a custom adaptor! Here's how:

1. Add your adaptor to `src/unicorn_eval/adaptors/`.
2. Inherit from one of the base adaptor classes in [`base.py`](src/unicorn_eval/adaptors/base.py).
3. Open a pull request with:
    - Your adaptor code
    - A short `README.md` that covers:
      - A clear description of your method
      - A list of tasks, or task types your method is designed for
    - A **unique name** (we will include your **team name** in the adaptor name to ensure you receive credit). When naming your method, please be **as specific as possible** — for example, indicate details like the number of layers or specific settings — so that related methods with different configurations can be distinctly named.
    - Any additional dependencies in a `requirements.txt` (details on adding new requirements below)

✅ Once accepted, your adaptor becomes selectable at submission time — and your team gets full recognition when it’s used!

> 💡 Keep in mind: we **prioritize originality**. If your adaptor is too similar to an existing one, it may not be accepted — so submit early and make it your own!

### Implementation requirements for contributing a new adaptor
- Your adaptor method must be implemented as a standalone function, following the baseline template [`base.py`](src/unicorn_eval/adaptors/base.py)
- It must complete within the allowed time limit of 1h
- It must run on CPU
- Submissions will be evaluated for correctness, efficiency, and compliance with the [challenge policies](https://unicorn.grand-challenge.org/requirements-and-guidelines/)
- 🚨 Important: Pre-trained adaptors are not allowed! Be original — you can use the few-shots, for example, for fitting or training your adaptor, but don’t rely on pre-trained solutions

### Dependencies
- Each method must be able to run in the [provided isolated environment](Dockerfile)
- Additional dependencies can be requested, but:
  - Approval of new dependencies is not guaranteed, dependencies will be evaluated based on compatibility with other packages
  - Organizers reserve the right to modify the list of dependencies over time, though we aim to maintain compatibility with existing adaptors
  - When specifying dependencies, use the least restrictive version (e.g., package>=1.0.0) to ensure flexibility

> 💬 Teams are encouraged to share ideas and discuss approaches on the [Grand Challenge forum](https://grand-challenge.org/forums/forum/unicorn-740/). Support and Q&A will also be available through the forum.


## 📤 Requesting New Adaptor Runs

You can request us to apply additional adaptors to your **existing vision submission** without impacting your submission limit.

### 📧 Submission Instructions

1. Go to your submission URL: `https://unicorn.grand-challenge.org/evaluation/<leaderboard-specific-number>/submissions/<your-submission-id>/`  
   **(Use only this format — not other links)**

2. For each submission that you want to rerun with a new adaptor, specify:
   - The full submission link<br>
     _Example:_ `https://unicorn.grand-challenge.org/evaluation/30/submissions/bc9b9fe2-1f8d-4b9e-af7b-0edb87b127a4/`
   - The new adaptor(s) you want to apply (chosen from the [Supported Adaptors](src/unicorn_eval/adaptors/README.md)).<br>
    ⚠️ Responsible use: You’re welcome to submit additional adaptor run requests over time. However, to ensure fair access for all participants, we ask that each request remains targeted and minimal (e.g., max 2 adaptors per leaderboard per request). Bulk or unfocused requests may be deprioritized.

3. Email your request to `support@unicorn-challenge.com` containing the following template:

```Subject: UNICORN Adaptor Run Request
Submission: https://unicorn.grand-challenge.org/evaluation/<leaderboard-specific-number>/submissions/your-submission-id/
Adaptors:
- teamname_adaptorX_v1
- teamname_adaptorY_v2

[Repeat for other submissions if needed]
```

## Summary

| **Modality**         | **What You Submit**                        | **Are Adaptors Used?** | **Submission Limit Applies To** |
|-----------------------|--------------------------------------------|-------------------------|-----------------------------------|
| 👁️ **Vision**            | Algorithm (feature extractor) + Adaptor   | ✅ Yes                  | Algorithm only                   |
| 🧠 **Language**          | Algorithm (predictive)                    | ❌ No                   | Algorithm                        |
| 🧠 **Vision-Language**   | Algorithm (predictive)                    | ❌ No                   | Algorithm                        |
