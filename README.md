# Esophageal-Cancer-Model-Code
## Introduction
This code repository focuses on the development and evaluation of prognostic prediction models for esophageal cancer, mainly centering on key prognostic indicators such as Overall Survival (OS). It implements a complete process from data preprocessing, model training to performance evaluation. Core functions include a custom loss function model based on CatBoost, ensemble learning sampling strategies, Cox proportional hazards model survival analysis, active learning sample selection, and decision curve analysis (DCA) and other clinical utility assessment tools.
## Environment
**Python 3.7+** and **R 4.3.3**
## Data
Our retrospective study was approved by the ethics committee (2022KY206). After data inclusion and exclusion screening, image reconstruction and segmentation, as well as feature value extraction, data preprocessing was carried out. Finally, 490 cases of data were retained, and 99 of them were used as internal validation data. Subsequently, an active learning strategy was applied to select 370 samples with high heterogeneity from the remaining 391 cases of data, completing the selection of the training set.
## Code Structure
| Name               | Function                                                                 |
|------------------------|--------------------------------------------------------------------------|
| Ensemble Sampling.py   | Integrate multiple classifiers (CatBoost / Random Forest / SVM) to achieve sample sampling and training with labeled noise |
| final Catboost model.py | Core model: CatBoost custom loss function + PFS data preprocessing + hyperparameter tuning + visualization |
| test survival.py       | Cox proportional hazards model training + feature selection + survival analysis (consistency index / survival curve) |
| Active Learning.py     | Based on the active learning strategy of BALD, iteratively select high-value samples to expand the training set |
| DCA and calibration.py | Decision curve analysis (DCA) + model calibration curve, to evaluate the clinical net benefit of the model  |
