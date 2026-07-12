# SAM 3 Surgical Video Segmentation

Text-prompted video segmentation and temporal tracking of laparoscopic surgery
using Meta's **Segment Anything Model 3 (SAM 3)** and its Promptable Concept
Segmentation (PCS) capability, benchmarked on the **CholecSeg8k** dataset.

> **Status:** work in progress. Module 0 complete.

## Motivation

SAM 3 performs open-vocabulary segmentation from short noun-phrase prompts.
It was not trained on surgical imagery. This project asks: **how well does that
zero-shot capability transfer to laparoscopic video**, where the "concepts" are
organs and surgical instruments, and where mistakes matter?

## Tech stack

Python 3.12 · PyTorch · SAM 3 · OpenCV · NumPy

## Setup

```bash
git clone https://github.com/shivamaiprojects/SAM_project.git
cd SAM_project
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python scripts/check_env.py
```

