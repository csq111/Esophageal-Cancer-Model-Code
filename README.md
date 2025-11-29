# Esophageal-Cancer-Model-Code
## Introduction
This code repository focuses on the development and evaluation of prognostic prediction models for esophageal cancer, mainly centering on key prognostic indicators such as Overall Survival (OS). It implements a complete process from data preprocessing, model training to performance evaluation. Core functions include a custom loss function model based on CatBoost, ensemble learning sampling strategies, Cox proportional hazards model survival analysis, active learning sample selection, and decision curve analysis (DCA) and other clinical utility assessment tools.
## Environment
**Python 3.7+** and **R 4.3.3**
## Data
The dataset used in the code consists of clinical characteristics and prognosis data of patients with esophageal cancer, and mainly includes the following types: 
- Baseline characteristics (such as age, tumor location, TNM stage, etc.)
- Treatment-related characteristics (such as radiotherapy dose, treatment plan, etc.)
- Prognostic indicators (**OS: overall survival**; **PFS: progression-free survival**; **LRFS: no local recurrence survival**; **LC: local control**, etc.) 

The path of the data file is given as a local path example in the code. When using it in practice, it should be replaced with your own dataset path, and ensure that the data format is consistent with the processing logic in the code (CSV format, including feature columns and prognosis label columns).
Our retrospective study was approved by the ethics committee **(2022KY206)**. After data inclusion and exclusion screening, image reconstruction and segmentation, as well as feature value extraction, data preprocessing was carried out. Finally, **490** cases of data were retained, and **99** of them were used as internal validation data. Subsequently, an active learning strategy was applied to select **370** samples with high heterogeneity from the remaining 391 cases of data, completing the selection of the training set.
## Code Structure
| Name               | Function                                                                 |
|------------------------|--------------------------------------------------------------------------|
| Ensemble Sampling.py   | Integrate multiple classifiers (CatBoost / Random Forest / SVM) to achieve sample sampling and training with labeled noise |
| final Catboost model.py | Core model: CatBoost custom loss function + PFS data preprocessing + hyperparameter tuning + visualization |
| test survival.py       | Cox proportional hazards model training + feature selection + survival analysis (consistency index / survival curve) |
| Active Learning.py     | Based on the active learning strategy of BALD, iteratively select high-value samples to expand the training set |
| DCA and calibration.py | Decision curve analysis (DCA) + model calibration curve, to evaluate the clinical net benefit of the model  |
| forest plot.R | Draw forest plots and other charts based on the existing data |
| km curve.R | Draw the km curve based on the Python running result |
## Guide
**1. Data Preparation**
   The `sample_data` folder in the project root directory already contains simulated clinical data for esophageal cancer (de-identified data to protect privacy). File structure:
   **Key Fields**:
- `Age` (Age)
- `Location` (Tumor Location)
- `TNM` (Staging)
- `PFS` (Progression-Free Survival)
- `PFS_m` (Progression-Free Time, Months)
**2. Core Step**
   Step1: Run Activate Learning.py. For activately selecting the data with high heterogeneity.
   Step2: Run final Catboost model.py to build and evaluate the model
  
  
## Notes
1. Data path: The local absolute path is used in the code. During actual deployment, it is necessary to modify it to a relative path or dynamic path according to the location of the data set.
2. Feature processing: The features corresponding to different prognostic indicators (OS/PFS/LRFS/LC) may vary. The feature selection logic needs to be adjusted according to the specific task.
3. Hyperparameter tuning: The depth of the model, the number of iterations, etc. are hyperparameters that need to be tuned based on the actual data set (refer to the GridSearchCV example in the code).
4. Custom loss: The `LoglossObjective` class supports multiple loss fusion. The penalty coefficient (penalty), reward factor (reward_factor), and other parameters can be adjusted according to the task requirements. 

## Expansion Directions
- Conduct more comparative experiments with various machine learning models (such as LightGBM, neural networks).
- Improve the analysis of model interpretability (such as SHAP value calculation).
- Optimize the active learning strategy by integrating uncertainty quantification with clinical prior knowledge. 

## Contact Information
If you have any questions, please contact **shiqingleicmu@gmail.com** and 223050027@link.cuhk.edu.cn.
