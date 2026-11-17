# Flood Detector-Training and evaluation pipeline for edge deployment

This repository contains the official implementation of the deep learning pipeline, data augmentation strategies, and uncertainty quantification framework proposed in the article: **"An Image-Based Urban Flood Detection System using Deep Learning and Conformal Prediction for Edge Deployment"**.

## Overview

Standard neural networks typically produce overconfident deterministic predictions, which constitutes a critical vulnerability in environmental disaster monitoring. To address this limitation, our proposed system is designed for edge-computing environments (e.g., IoT cameras) to provide risk-averse early warnings for urban floods. The pipeline integrates four core methodological components:

* **MobileNetV3Large:** A lightweight, edge-native architecture optimized via two-phase fine-tuning.
* **Dynamic Augmentation and Semantic Masking:** Class-conditional environmental distortions (e.g., simulated rain, fog, and shadows) applied to mitigate spatial overfitting on critical classes.
* **Sparse Categorical Focal Loss ($\gamma = 2.0$):** A mathematically weighted loss function designed to aggressively suppress the gradient contribution of the severely imbalanced majority class (normal water levels).
* **Class-Conditional Conformal Prediction (CP):** A statistical framework that bounds predictive uncertainty. When extreme visual noise forces an empty prediction set ($\emptyset$), a risk-averse fallback logic systematically defaults to the argmax prediction, strictly prioritizing false alarms over fatal false negatives.

## Repository Structure

* `flood-detector-training-and-evaluation-pipeline.ipynb`: The comprehensive Jupyter Notebook containing the data preprocessing, model training, Focal Loss formulation, and Conformal Prediction calibration.
* `original_training_report.pdf`: The official evaluation report documenting the exact performance metrics and confusion matrix published in the manuscript.

## Reproducibility Notice

Deep learning models trained on GPUs exhibit inherent non-determinism due to parallel floating-point operations. Re-running the provided notebook from scratch will result in slight stochastic metric variations ($\pm$ 1-2%) compared to the exact numbers reported in the official manuscript.

The architectural choices, the Sparse Categorical Focal Loss implementation, and the Conformal Prediction fallback logic provided in the notebook remain perfectly consistent with the paper. The exact metrics (e.g., Global Accuracy: 96.48%, Flood F1-Score: 78.65%) validated during our research are preserved in the `original_training_report.pdf` provided in this repository.

## Usage and Reproduction

To reproduce the pipeline without local hardware constraints, the codebase is designed to be executed directly via Kaggle. The pipeline strictly requires two specific public datasets hosted on the platform:

1. **River Images Dataset:** Contains the raw camera frames and the annotation file. [https://www.kaggle.com/datasets/caetanoranieri/river-images-at-sao-carlos]
2. **River Masks Dataset:** Contains the semantic segmentation masks utilized for the dynamic background augmentation. [https://www.kaggle.com/datasets/guilhermebisse/sao-carlos-river-masks-full-batch]

**Execution Steps:**
1. Download the `flood_detection_pipeline.ipynb` file from this repository.
2. Import the notebook into a new Kaggle environment.
3. Use the "Add Data" feature in the Kaggle interface to attach both datasets listed above to your environment.
4. Ensure that the `DATASET_IMAGENS`, `DATASET_MASCARAS`, and `CSV_PATH` variables in the configuration cell match the correct Kaggle input directories.
5. Execute all cells to run the training pipeline and generate the automated evaluation report and conformal prediction visual analytics.

## Citation

If you utilize this code or our methodology in your research, please cite our paper:

```bibtex
@inproceedings{rodrigues2026image,
  title={An Image-Based Urban Flood Detection System using Deep Learning and Conformal Prediction for Edge Deployment},
  author={Rodrigues, Guilherme Bisse Alves and Rocha, Arthur Lima Marques and Ueyama, J{\'o} and Ranieri, Caetano Mazzoni},
  booktitle={Encontro Nacional de Intelig{\^e}ncia Artificial e Computacional (ENIAC)},
  pages={2655--2666},
  year={2026},
  organization={SBC}
}
