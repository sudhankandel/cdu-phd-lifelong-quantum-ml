
# CDU PhD — Lifelong Learning and Quantum Machine Learning

Implementation of the Charles Darwin University (CDU) PhD assessment on **Lifelong Learning and Quantum Machine Learning**, using the ISIC 2018 skin lesion classification dataset.

This project investigates catastrophic forgetting in sequential class-incremental learning and compares classical machine learning models with a hybrid quantum-classical variational quantum circuit (VQC) approach.

## 1. Project Overview

The experiments investigate three main research questions:

1. How does sequential learning affect classification performance and catastrophic forgetting?
2. Can rehearsal-based memory reduce forgetting across sequential learning tasks?
3. How does a hybrid quantum-classical classifier compare with classical classifiers under different memory budgets and finite-shot quantum measurements?

The implementation includes:

- Classical classification head
- Parameter-matched classical head
- Classical MLP baseline
- Hybrid quantum-classical VQC classifier
- Naive sequential learning
- Rehearsal-based continual learning
- Memory budgets of 5%, 10%, 20%, and 100%
- Validation-based hyperparameter tuning
- Finite-shot quantum experiments
- Balanced accuracy and forgetting evaluation
- Exploratory data analysis (EDA)
- Reproducible experiment configurations and results

## 2. Dataset

**Dataset:** ISIC Challenge 2018 — Task 3

Official source:

https://challenge.isic-archive.com/data/#2018

The experiments use six skin lesion categories:

| Label | Description |
|-------|-------------|
| DF | Dermatofibroma |
| AKIEC | Actinic keratoses and intraepithelial carcinoma |
| BKL | Benign keratosis-like lesions |
| NV | Melanocytic nevi |
| VASC | Vascular lesions |
| BCC | Basal cell carcinoma |

The ISIC 2018 dataset also contains MEL (melanoma), which is not included in this six-class experimental configuration.

### Data Preparation

The dataset preparation workflow includes:

- Image and label validation
- Class distribution analysis
- Image preprocessing
- Stratified 80/20 train-test splitting
- Training-derived validation splitting
- Sequential class-incremental task construction
- Reproducible random seed configuration

The test set remains separate from validation-based model selection.

### Reproducibility Seed

The random seed is derived from the applicant reference number.

**Seed: 4826**

```bash
export CDU_SEED=4826
python data_preparation.py --labels data/labels.csv
```

For Windows PowerShell:

```powershell
$env:CDU_SEED="4826"
python data_preparation.py --labels data/labels.csv
```

The same seed is used throughout the final experimental workflow.

## 3. Project Structure

```text
cdu-phd-lifelong-quantum-ml/
│
├── data/
│   ├── labels.csv
│   └── feature_cache/
│
├── results/
│
├── analysis.ipynb
├── data_preparation.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Main Components

| File | Purpose |
|------|---------|
| data_preparation.py | Dataset validation, splitting and preparation |
| train.py | Training, tuning and evaluation |
| analysis.ipynb | Exploratory data analysis and visualisation |
| requirements.txt | Python dependencies |
| results/ | Experiment outputs, metrics and logs |
| README.md | Project documentation |

Additional result files and metadata may be generated during execution.

## 4. Installation

### Clone the Repository

```bash
git clone <repository-url>
cd cdu-phd-lifelong-quantum-ml
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To record the exact tested environment:

```bash
pip freeze > environment-lock.txt
```

For reproducibility, use the dependency versions associated with the submitted experiments.

## 5. Feature Extraction

A pretrained **ResNet-18** backbone is used for image feature extraction.

Configuration:

- Backbone: ResNet-18
- Pretraining: ImageNet-1K
- Input resolution: 224 × 224
- Feature dimension: 512
- Cached feature representations for repeated experiments

The feature cache reduces repeated image-processing overhead and allows classical and quantum heads to be compared using the same extracted representations.

Feature caches are stored under:

```text
data/feature_cache/
```

## 6. Continual Learning

The dataset is organised into sequential class-incremental tasks.

Models learn tasks in sequence rather than training on all classes simultaneously.

### Naive Learning

The model learns each task without retaining examples from earlier tasks.

This provides a baseline for measuring catastrophic forgetting.

### Rehearsal Learning

A portion of previously observed training samples is retained and replayed during subsequent tasks.

The experiments investigate:

| Memory | Description |
|--------|-------------|
| 0% | No rehearsal |
| 5% | Small replay memory |
| 10% | Moderate replay memory |
| 20% | Larger replay memory |
| 100% | Full replay reference |

Rehearsal is intended to reduce forgetting by preserving information about previously learned classes.

## 7. Models

### Classical Classifier

A classical classification head operates on the ResNet-18 feature representations.

### Parameter-Matched Classical Classifier

A classical head with a parameter budget designed for comparison with the quantum model.

This provides a more controlled comparison between classical and quantum approaches.

### Classical MLP

An additional multilayer perceptron baseline.

### Hybrid Quantum-Classical Classifier

The quantum model combines:

1. Classical image feature extraction
2. Classical-to-quantum feature transformation
3. Variational quantum circuit
4. Quantum measurements
5. Classical output classification

The variational quantum circuit uses trainable parameters optimised during model training.

## 8. Running Experiments

### Classical Naive Baseline

```bash
python train.py \
  --model classical \
  --strategy naive \
  --eval-mode validation
```

### Classical Rehearsal — 10%

```bash
python train.py \
  --model classical \
  --strategy rehearsal \
  --memory 10 \
  --eval-mode validation
```

### Quantum Naive Baseline

```bash
python train.py \
  --model quantum \
  --strategy naive \
  --eval-mode validation
```

### Quantum Rehearsal — 10%

```bash
python train.py \
  --model quantum \
  --strategy rehearsal \
  --memory 10 \
  --eval-mode validation
```

### Parameter-Matched Classical Model

```bash
python train.py \
  --model classical_matched \
  --strategy rehearsal \
  --memory 10 \
  --eval-mode validation
```

## 9. Hyperparameter Tuning

Hyperparameter tuning is performed using the training-derived validation set.

The final test set must not be used for selecting hyperparameters.

### Search Space

| Hyperparameter | Values |
|----------------|--------|
| Learning rate | 0.0001, 0.0005, 0.001 |
| Epochs | 10, 20 |
| Batch size | 8 |

The selection criterion is the final average validation balanced accuracy.

The best validation configuration is saved and reused for final evaluation.

### Final Evaluation with Best Configuration

```bash
python train.py \
  --model classical \
  --strategy rehearsal \
  --memory 10 \
  --eval-mode test \
  --use-best-config
```

This loads the previously selected configuration for the specified experiment.

## 10. Memory Budget Experiments

### Classical Model

```bash
python train.py --model classical --strategy rehearsal --memory 5 --eval-mode test --use-best-config

python train.py --model classical --strategy rehearsal --memory 10 --eval-mode test --use-best-config

python train.py --model classical --strategy rehearsal --memory 20 --eval-mode test --use-best-config

python train.py --model classical --strategy rehearsal --memory 100 --eval-mode test --use-best-config
```

### Quantum Model

```bash
python train.py --model quantum --strategy rehearsal --memory 5 --eval-mode test --use-best-config

python train.py --model quantum --strategy rehearsal --memory 10 --eval-mode test --use-best-config

python train.py --model quantum --strategy rehearsal --memory 20 --eval-mode test --use-best-config

python train.py --model quantum --strategy rehearsal --memory 100 --eval-mode test --use-best-config
```

Run each configuration only after its corresponding validation-based hyperparameter selection is complete.

## 11. Finite-Shot Quantum Experiments

Quantum measurements are evaluated using a finite number of shots.

The finite-shot experiment investigates the effect of measurement sampling on classification performance.

### Quantum Model — 1,024 Shots

```bash
python train.py \
  --model quantum \
  --strategy rehearsal \
  --memory 20 \
  --eval-mode test \
  --lr 0.0001 \
  --epochs 20 \
  --batch-size 8 \
  --shots 1024
```

Finite-shot experiments should use comparable configurations and training budgets when comparing against exact-expectation quantum simulations.

## 12. Evaluation Metrics

### Balanced Accuracy

Balanced accuracy is the mean recall across classes:

```text
Balanced Accuracy =
    (1 / C) × sum(Recall for each class)
```

It is particularly relevant for the imbalanced ISIC dataset.

### Accuracy Matrix

The continual-learning accuracy matrix records model performance on each task after successive training stages.

```text
A[i,j] = Accuracy on task j after training task i
```

### Average Accuracy

Average accuracy measures performance across the tasks learned up to a given stage.

### Average Forgetting

Forgetting measures the reduction in performance on earlier tasks after learning subsequent tasks.

### Additional Evaluation

Where available, experiments also record:

- Per-class recall
- Confusion matrices
- Training and validation metrics
- Experiment configurations
- Training logs
- Runtime information

## 13. Exploratory Data Analysis

The project includes an EDA notebook:

```text
analysis.ipynb
```

The notebook is used to investigate:

- Dataset size
- Class distribution
- Class imbalance
- Sample images
- Image characteristics
- Training and testing distributions
- Data preparation outputs

Launch Jupyter Notebook:

```bash
jupyter notebook analysis.ipynb
```

EDA visualisations can be included in the Task 1 report to explain dataset characteristics and justify the evaluation methodology.

## 14. Experimental Results

Experiment outputs are stored in:

```text
results/
```

The results directory is intended to preserve the generated evaluation artifacts, such as:

- Model performance metrics
- Accuracy matrices
- Forgetting measurements
- Training logs
- Hyperparameter configurations
- Experiment summaries
- Visualisations

Numerical results should be taken directly from the generated experiment files.

No performance figures are hardcoded into this README because reported values must correspond to verified experimental outputs.

## 15. Reproducibility

The project is designed to support reproducible experiments.

Important controls include:

- Fixed random seed: 4826
- Stratified dataset splitting
- Separate training, validation and test data
- Shared ResNet-18 feature extractor
- Cached feature representations
- Validation-based hyperparameter selection
- Recorded model configurations
- Consistent evaluation metrics
- Saved experiment results
- Git version history
## 19. References

1. ISIC Challenge 2018  
   https://challenge.isic-archive.com/data/#2018

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.

3. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.

4. Schuld, M., & Killoran, N. (2019). Quantum Machine Learning in Feature Hilbert Spaces. Physical Review Letters.

---

**Dataset:** ISIC 2018 Task 3  
**Random Seed:** 4826  
**Framework:** Python, PyTorch and Quantum Machine Learning  
**Status:** Experimental implementation and evaluation
