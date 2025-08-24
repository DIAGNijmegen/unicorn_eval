# Supported Adaptors

Below are the approved adaptors for each task type.

---

### 🧩 Adaptor Pathology Classification

- `1-nn`
- `5-nn`
- `20-nn`
- `1-nn-weighted`
- `5-nn-weighted`
- `20-nn-weighted`
- `logistic-regression`
- `linear-probing`
- `mlp`

---

### 🧩 Adaptor Pathology Segmentation

- `segmentation-upsampling`

---

### 🧩 Adaptor Pathology Detection

- `density-map`
- `conv-detector` contributed by [MEVIS-M3](https://unicorn.grand-challenge.org/teams/4756/)

---

### 🧩 Adaptor Pathology Regression

- `1-nn`
- `5-nn`
- `20-nn`
- `1-nn-weighted`
- `5-nn-weighted`
- `20-nn-weighted`
- `linear-probing`
- `linear-probing-survival`
- `mlp`
- `mlp-survival`

---

### 🧩 Adaptor Radiology Classification

- `1-nn-weighted`
- `5-nn-weighted`
- `20-nn-weighted`
- `logistic-regression`
- `linear-probing`
- `mlp`

---

### 🧩 Adaptor Radiology Segmentation

- `segmentation-upsampling-3d` supported by [AIAH LAB](https://unicorn.grand-challenge.org/teams/4863/)
- `linear-upsample-conv3d` contributed by [AIMHI](https://unicorn.grand-challenge.org/teams/4707/)
- `linear-upsample-conv3d-v2` adapted from `linear-upsample-conv3d` by the organizers.
- `conv3d-linear-upsample` contributed by [AIMHI](https://unicorn.grand-challenge.org/teams/4707/)
- `conv-segmentation-3d` contributed by [MEVIS-M3](https://unicorn.grand-challenge.org/teams/4756/)

---

### 🧩 Adaptor Radiology Detection Points

- `patch-nodule-regressor`

---

### 🧩 Adaptor Radiology Detection Segmentation

- `detection-by-segmentation-upsampling-3d`
- `conv-detection-segmentation-3d` contributed by [MEVIS-M3](https://unicorn.grand-challenge.org/teams/4756/)
- `detection-by-linear-upsample-conv3d` contributed by [AIMHI](https://unicorn.grand-challenge.org/teams/4707/)
- `detection-by-conv3d-linear-upsample` contributed by [AIMHI](https://unicorn.grand-challenge.org/teams/4707/)