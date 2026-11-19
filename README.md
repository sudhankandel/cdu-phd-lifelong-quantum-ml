# CDU PhD — Lifelong Learning and Quantum ML

Starter repository for the CDU PhD assessment using ISIC 2018 Task 3 skin lesion images.

**Status:** Project scaffold only. Training, rehearsal, quantum VQC and finite-shot
experiments are **not yet implemented**. Do not report results until actually run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After verifying compatibility, record exact installed versions in a lock/snapshot
(e.g. `pip freeze > environment-lock.txt`) and commit that snapshot. Keep
`requirements.txt` aligned with the tested environment.

## Dataset

Download **ISIC Challenge 2018 Task 3** images and ground truth from:
https://challenge.isic-archive.com/data/#2018

Prepare `data/labels.csv` with exactly two columns: `image,label`, one image per
row, with the seven official disease class labels. Do not commit raw image data.
Record source files and preprocessing choices in the final README.

## Seed and split

Use the **last four digits of your applicant reference number** (not student ID).
On macOS/Linux:

```bash
export CDU_SEED=1234  # REPLACE with your real last four digits
python data_preparation.py --labels data/labels.csv
```

On PowerShell: `$env:CDU_SEED='1234'` (replace with your own seed).
The split metadata and CSV files are written to `data/`. For final
reproducibility, preserve the metadata and split manifests as tracked artifacts
in a suitable location, without committing the full image dataset.

## Planned experiment interface (not implemented yet)

```bash
python train.py --model classical --strategy naive
python train.py --model quantum --strategy rehearsal --memory 10
python train.py --model quantum --strategy rehearsal --memory 20 --shots 1024
```

These commands currently validate arguments and intentionally raise
`NotImplementedError` until training is implemented.

## Required deliverables

- Full Git commit history
- Training logs and experiment configurations
- Exact, tested dependency versions and run instructions
- Reproducible task sequence and 80/20 stratified split
- Balanced-accuracy matrices, average accuracy, average forgetting
- Naive, rehearsal (5/10/20/100%), matched-head and 1,024-shot experiments
- Candidate-written Q1 report and Q2 two-page proposal

**Assessment restriction:** Generative AI must not write the report or proposal.
Declare any AI coding assistance at the end of your own report.

**Confidentiality:** Keep this repository private unless authorised otherwise.
