import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import KaplanMeierFitter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tabulate import tabulate
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# ------------------------------------------------------------------------------
# Configuration Class: Manage all paths, hyperparameters and settings
# ------------------------------------------------------------------------------
class ModelConfig:
    """
    Configuration for data paths, model hyperparameters and visualization settings
    Replace the placeholder paths with your actual file paths
    """
    # Data file paths
    DATA_PATHS = {
        "combined_OS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data_OS.csv",
        "combined_LC": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data LC.csv",
        "combined_PFS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data PFS.csv",
        "combined_LRFS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data LRFS.csv",
        "after_OS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New OS.csv",
        "after_LC": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - LC.csv",
        "after_PFS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - PFS.csv",
        "after_LRFS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - LRFS.csv",
        "samples_420": r"D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples.csv",
        "moved_samples": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after.csv",
        "before_420_OS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master/before 420 samples OS 7.12.csv",
        "before_moved_OS": r"D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before OS 7.12.csv",
        "final_OS_before": r"D:\Acsq\BIM硕士第三学期\AD/before OS 7.19.csv",
        "final_LRFS": r"D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 LRFS.csv",
        "final_OS_450": r"D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_labeled_samples 370.csv",
        "final_OS_42": r"D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_unlabeled_samples 21 7.21.csv",
        "test_data": r"D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment test data.csv",
        "before_LRFS": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment LRFS data.csv",
        "train_LRFS_labeled": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.1 LRFS.csv",
        "test_LRFS_unlabeled": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.1 LRFS.csv",
        "before_PFS": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment-8.2 PFS.csv",
        "before_LC": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment-8.2 LC.csv",
        "before_OS": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 OS.csv"
    }

    # CatBoost hyperparameters
    CATBOOST_PARAMS = {
        "loss_function": None,  # Will be set to custom LoglossObjective
        "eval_metric": "Logloss",
        "cat_features": ["Age", "Location", "N", "TNM", "PTV_Dose", "GTV_Dose", "ECOG", "T", "Chemotherapy"],
        "random_seed": 42,
        "early_stopping_rounds": 50
    }

    # Preprocessing settings
    OUTLIER_Z_THRESHOLD = 4
    CORR_THRESHOLD = 0.8
    EXTERNAL_VALIDATION_SIZE = 60
    TEST_SIZE = 0.3
    STANDARDIZE_KEYWORDS = ["treatment", "TL", "original"]
    NOISE_LEVEL = 0.08

    # Hyperparameter search grid
    PARAM_GRID = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.1],
        "iterations": [400, 600, 800, 1000]
    }

    # Visualization settings
    PLOT_FIGSIZE = (8, 6)
    LARGE_FIGSIZE = (12, 8)
    TITLE_FONTSIZE = 25
    LABEL_FONTSIZE = 20
    KM_FONTSIZE = 27
    TOP_FEATURES = 10

# ------------------------------------------------------------------------------
# Custom Loss Function for CatBoost
# ------------------------------------------------------------------------------
class CustomLoglossObjective:
    """
    Custom Logloss Objective for CatBoost with penalty/reward mechanism
    Combines Logloss, Quantile Loss, Focal Loss and Huber Loss derivatives
    Penalty for overconfident wrong predictions, reward for confident correct predictions
    """
    def __init__(self, penalty=2, alpha=0.5, mrf_weight=0.5, reward_factor=0.9):
        self.alpha = alpha
        self.penalty = penalty
        self.mrf_weight = mrf_weight
        self.reward_factor = reward_factor

    def stable_log(self, x, eps=1e-10):
        """Numerically stable log calculation to avoid division by zero"""
        return np.log1p(x + eps) if x > 1 - eps else np.log(x + eps)

    def calc_ders_range(self, approxes, targets, weights):
        """
        Calculate first and second derivatives for CatBoost custom loss function
        Args:
            approxes (list): Model predictions (log-odds)
            targets (list): True binary labels (0/1)
            weights (list): Sample weights (None if not provided)
        Returns:
            list: Pairs of (first derivative, second derivative) for each sample
        """
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = [np.exp(a) for a in approxes]
        gamma = 2  # Focal loss gamma
        beta = 0.5  # Focal loss beta
        delta = 0.1  # Huber loss delta
        q = 0.6  # Quantile loss quantile
        result = []

        for idx in range(len(targets)):
            # Probability from log-odds
            p = exponents[idx] / (1 + exponents[idx])
            
            # Dynamic penalty for overconfident wrong predictions
            penalty = 1.0
            if (targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8):
                penalty *= self.penalty

            # Dynamic reward for confident correct predictions
            reward_factor = 1.0
            if (targets[idx] == 1 and p > 0.8) or (targets[idx] == 0 and p < 0.2):
                reward_factor *= self.reward_factor

            # Logloss derivatives
            der1_log = (1 - p) * self.penalty if targets[idx] > 0.0 else -p * self.penalty
            der2_log = -p * (1 - p)

            # Quantile loss derivatives
            der1_quantile = q * (p - p**2) if targets[idx] - p >= 0 else (q - 1) * (p - p**2)
            der2_quantile = 0

            # Combine Logloss and Quantile loss derivatives
            der1_combined = (der1_log + der1_quantile) / 2
            der2_combined = (der2_log + der2_quantile) / 2

            # Apply sample weights if provided
            if weights is not None:
                der1_combined *= weights[idx]
                der2_combined *= weights[idx]

            result.append((der1_combined, der2_combined))

        return result

# ------------------------------------------------------------------------------
# Data Preprocessing Functions
# ------------------------------------------------------------------------------
def load_all_data(config):
    """
    Load all datasets from the configuration paths and drop missing values
    Args:
        config (ModelConfig): Configuration object with data paths
    Returns:
        dict: Dictionary of loaded DataFrames
    Raises:
        FileNotFoundError: If any data file path is invalid
    """
    data_dict = {}
    for key, path in config.DATA_PATHS.items():
        try:
            data_dict[key] = pd.read_csv(path).dropna()
            print(f"Loaded {key} dataset with {len(data_dict[key])} samples")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
    
    # Special processing for test_LRFS_unlabeled (drop LRFS_m column)
    if "test_LRFS_unlabeled" in data_dict:
        data_dict["test_LRFS_unlabeled"] = data_dict["test_LRFS_unlabeled"].drop(["LRFS_m"], axis=1)
    
    return data_dict

def add_noise_to_labels(y, noise_level):
    """
    Add random noise to binary labels by flipping a percentage of samples
    Args:
        y (np.ndarray): Original binary labels (0/1)
        noise_level (float): Fraction of labels to flip (0 to 1)
    Returns:
        np.ndarray: Noisy labels
    """
    num_samples = len(y)
    num_noise = int(noise_level * num_samples)
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    
    noisy_y = np.copy(y)
    noisy_y[noise_indices] = 1 - noisy_y[noise_indices]
    return noisy_y

def preprocess_pfs_data(before_data_PFS, config):
    """
    Preprocess PFS data: remove outliers, drop highly correlated features, split features/labels
    Args:
        before_data_PFS (pd.DataFrame): Raw PFS dataset
        config (ModelConfig): Preprocessing settings
    Returns:
        tuple: (X_final, X, y, outside_validation)
    """
    # Step 1: Remove outliers using Z-score
    z_score = stats.zscore(before_data_PFS)
    outliers = (z_score > config.OUTLIER_Z_THRESHOLD).any(axis=1)
    data_no_outliers = before_data_PFS[~outliers].dropna()
    print(f"Data after outlier removal: {len(data_no_outliers)} samples")

    # Step 2: Drop highly correlated features
    corr_matrix = data_no_outliers.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > config.CORR_THRESHOLD)]
    X_final = data_no_outliers.drop(to_drop, axis=1)
    print(f"Data after dropping correlated features: {X_final.shape[1]} features remaining")

    # Step 3: Sample external validation set
    outside_validation = X_final.sample(n=config.EXTERNAL_VALIDATION_SIZE, random_state=config.CATBOOST_PARAMS["random_seed"])
    print(f"External validation set sampled: {len(outside_validation)} samples")

    # Step 4: Split features and labels
    X = X_final.drop(["PFS_m", "PFS"], axis=1).astype(str)
    y = X_final["PFS"].values

    return X_final, X, y, outside_validation

def standardize_features(X, config):
    """
    Standardize features containing specific keywords using StandardScaler
    Args:
        X (pd.DataFrame): Feature DataFrame
        config (ModelConfig): Standardization settings
    Returns:
        pd.DataFrame: Standardized feature DataFrame
    """
    scaler = StandardScaler()
    columns_to_standardize = [col for col in X.columns if any(kw in col for kw in config.STANDARDIZE_KEYWORDS)]
    
    if columns_to_standardize:
        X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
        print(f"Standardized columns: {columns_to_standardize}")
    return X

# ------------------------------------------------------------------------------
# Model Training and Evaluation Functions
# ------------------------------------------------------------------------------
def train_catboost_model(X_train, X_test, y_train, y_test, config):
    """
    Train CatBoost model with custom loss function
    Args:
        X_train/X_test (pd.DataFrame): Training/test features
        y_train/y_test (np.ndarray): Training/test labels
        config (ModelConfig): CatBoost hyperparameters
    Returns:
        tuple: (catboost_model, feature_importance, train_pool)
    """
    # Create CatBoost Pool objects
    train_pool = Pool(X_train, y_train, cat_features=config.CATBOOST_PARAMS["cat_features"])
    test_pool = Pool(X_test, y_test, cat_features=config.CATBOOST_PARAMS["cat_features"])

    # Initialize custom loss
    custom_loss = CustomLoglossObjective()
    config.CATBOOST_PARAMS["loss_function"] = custom_loss

    # Train model
    catboost_model = CatBoostClassifier(**config.CATBOOST_PARAMS)
    catboost_model.fit(train_pool, eval_set=test_pool, verbose=False)

    # Get feature importance
    feature_importance = catboost_model.get_feature_importance(data=train_pool, type="PredictionValuesChange")

    return catboost_model, feature_importance, train_pool

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate key classification metrics: Accuracy, Sensitivity, Specificity, Precision, F1-Score
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    Returns:
        tuple: (accuracy, sensitivity, specificity, precision, f1_score)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return accuracy, sensitivity, specificity, precision, f1_score

def evaluate_catboost_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate CatBoost model: predict labels/probabilities, calculate metrics and ROC-AUC
    Args:
        model (CatBoostClassifier): Trained CatBoost model
        X_train/X_test (pd.DataFrame): Training/test features
        y_train/y_test (np.ndarray): Training/test labels
    Returns:
        tuple: (metrics, train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc)
    """
    # Predict labels
    y_pred_test = model.predict(X_test)
    metrics = calculate_classification_metrics(y_test, y_pred_test)

    # Predict probabilities for ROC-AUC
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curves and AUC
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    # Print metrics
    print("\nCatBoost Classification Metrics:")
    print(f"Accuracy: {metrics[0]:.3f}")
    print(f"Sensitivity: {metrics[1]:.3f}")
    print(f"Specificity: {metrics[2]:.3f}")
    print(f"Precision: {metrics[3]:.3f}")
    print(f"F1 Score: {metrics[4]:.3f}")
    print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")

    return metrics, train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc

# ------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------
def plot_roc_curve(train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc, config):
    """
    Plot ROC curve for training and test sets
    Args:
        train_fpr/train_tpr (np.ndarray): Training ROC curve points
        test_fpr/test_tpr (np.ndarray): Test ROC curve points
        train_auc/test_auc (float): ROC-AUC scores
        config (ModelConfig): Visualization settings
    """
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(train_fpr, train_tpr, label=f"Train AUC = {train_auc:.3f}")
    plt.plot(test_fpr, test_tpr, label=f"Internal Validation AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate", fontsize=config.LABEL_FONTSIZE)
    plt.ylabel("True Positive Rate", fontsize=config.LABEL_FONTSIZE)
    plt.title("ROC Curve", fontsize=config.TITLE_FONTSIZE)
    plt.legend()
    plt.show()

def plot_learning_curve(model, config):
    """
    Plot CatBoost learning curve (train vs validation loss)
    Args:
        model (CatBoostClassifier): Trained CatBoost model
        config (ModelConfig): Visualization settings
    """
    evals_result = model.get_evals_result()
    train_loss = evals_result["learn"]["Logloss"]
    val_loss = evals_result["validation"]["Logloss"]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Internal Validation")
    plt.xlabel("Number of iterations", fontsize=config.LABEL_FONTSIZE)
    plt.ylabel("Loss Function", fontsize=config.LABEL_FONTSIZE)
    plt.title("CatBoost Learning Curve", fontsize=config.TITLE_FONTSIZE)
    plt.legend()
    plt.show()

def plot_km_survival(before_data_OS, before_data_PFS, config):
    """
    Plot Kaplan-Meier survival curves for OS and PFS
    Args:
        before_data_OS (pd.DataFrame): OS dataset with time and event columns
        before_data_PFS (pd.DataFrame): PFS dataset with time and event columns
        config (ModelConfig): Visualization settings
    """
    # Extract survival data
    durations_OS = before_data_OS["OS_m"].values
    events_OS = before_data_OS["OS"].values
    durations_PFS = before_data_PFS["PFS_m"].values
    events_PFS = before_data_PFS["PFS"].values

    # Fit Kaplan-Meier models
    kmf_os = KaplanMeierFitter()
    kmf_os.fit(durations_OS, event_observed=events_OS, label="OS")
    kmf_pfs = KaplanMeierFitter()
    kmf_pfs.fit(durations_PFS, event_observed=events_PFS, label="PFS")

    # Plot curves
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    kmf_os.plot(ax=ax)
    kmf_pfs.plot(ax=ax)
    plt.xlabel("Time", fontsize=config.KM_FONTSIZE)
    plt.ylabel("Survival probability", fontsize=config.KM_FONTSIZE)
    plt.title("Kaplan-Meier Curves of OS and PFS", fontsize=config.KM_FONTSIZE)
    plt.legend()
    plt.show()

def plot_top_features(feature_importance, train_pool, config):
    """
    Plot horizontal bar chart of top N feature importance from CatBoost
    Args:
        feature_importance (np.ndarray): CatBoost feature importance values
        train_pool (Pool): CatBoost training pool with feature names
        config (ModelConfig): Visualization settings
    """
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_indices = sorted_indices[:config.TOP_FEATURES]
    feature_names = train_pool.get_feature_names()
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importance, color="skyblue", align="center")
    plt.yticks(range(len(top_features)), top_features, fontdict={"size": config.LABEL_FONTSIZE})
    plt.xlabel("Importance", fontdict={"size": config.TITLE_FONTSIZE})
    plt.title(f"Top {config.TOP_FEATURES} Important Features", fontdict={"size": config.TITLE_FONTSIZE})
    plt.gca().invert_yaxis()
    plt.show()

# ------------------------------------------------------------------------------
# Hyperparameter Search Function
# ------------------------------------------------------------------------------
def catboost_parameter_search(X_train, y_train, X_test, y_test, config):
    """
    Grid search for CatBoost hyperparameters, evaluate metrics and plot top ROC curves
    Args:
        X_train/X_test (pd.DataFrame): Training/test features
        y_train/y_test (np.ndarray): Training/test labels
        config (ModelConfig): Hyperparameter grid and settings
    """
    evaluation_metrics = []
    roc_auc_values = []
    cat_features = config.CATBOOST_PARAMS["cat_features"]

    # Grid search over hyperparameters
    for depth in config.PARAM_GRID["depth"]:
        for lr in config.PARAM_GRID["learning_rate"]:
            for iterations in config.PARAM_GRID["iterations"]:
                # Initialize and train model
                model = CatBoostClassifier(
                    depth=depth,
                    learning_rate=lr,
                    iterations=iterations,
                    loss_function=CustomLoglossObjective(),
                    eval_metric="Logloss",
                    early_stopping_rounds=config.CATBOOST_PARAMS["early_stopping_rounds"],
                    cat_features=cat_features,
                    random_state=config.CATBOOST_PARAMS["random_seed"],
                    verbose=False
                )
                train_pool = Pool(X_train, y_train, cat_features=cat_features)
                test_pool = Pool(X_test, y_test, cat_features=cat_features)
                model.fit(train_pool, eval_set=test_pool)

                # Calculate ROC-AUC
                train_probs = model.predict_proba(X_train)[:, 1]
                test_probs = model.predict_proba(X_test)[:, 1]
                train_auc = auc(*roc_curve(y_train, train_probs)[:2])
                test_auc = auc(*roc_curve(y_test, test_probs)[:2])

                # Calculate classification metrics
                y_pred = (test_probs > 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
                npv = tn / (tn + fn) if (tn + fn) != 0 else 0
                icc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

                # Calculate 95% CI for AUC
                auc_ci = np.nan_to_num(test_auc)
                auc_se = np.sqrt(auc_ci * (1 - auc_ci) / len(y_test))
                auc_ci_lower, auc_ci_upper = stats.norm.interval(0.95, loc=auc_ci, scale=auc_se)

                # Store metrics
                evaluation_metrics.append({
                    "Depth": depth,
                    "Learning Rate": lr,
                    "Iterations": iterations,
                    "AUC": auc_ci,
                    "PPV": ppv,
                    "NPV": npv,
                    "ICC": icc,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "95% CI Lower": auc_ci_lower,
                    "95% CI Upper": auc_ci_upper,
                    "Train AUC": train_auc
                })
                roc_auc_values.append((depth, lr, iterations, test_auc))

    # Convert to DataFrame and sort
    metrics_df = pd.DataFrame(evaluation_metrics)
    metrics_df = metrics_df.sort_values(by="AUC", ascending=False)
    print("\nHyperparameter Search Results (Top 10):")
    print(tabulate(metrics_df.head(10), headers="keys", tablefmt="psql"))

    # Get top 5 parameter combinations by test AUC
    roc_auc_values = sorted(roc_auc_values, key=lambda x: x[3], reverse=True)[:5]

    # Plot ROC curves for top combinations (Train set)
    plt.figure(figsize=config.LARGE_FIGSIZE)
    for depth, lr, iterations, test_auc in roc_auc_values:
        model = CatBoostClassifier(depth=depth, learning_rate=lr, iterations=iterations, cat_features=cat_features)
        model.fit(X_train, y_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, train_probs)
        train_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Depth: {depth}, LR: {lr}, Iter: {iterations}\n(Train AUC = {train_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=config.TITLE_FONTSIZE)
    plt.ylabel("True Positive Rate", fontsize=config.TITLE_FONTSIZE)
    plt.title("ROC Curves for Top 5 Parameter Combinations on Train Set", fontsize=20)
    plt.legend(loc="lower right", fontsize="large")
    plt.show()

    # Plot ROC curves for top combinations (Test set)
    plt.figure(figsize=config.LARGE_FIGSIZE)
    for depth, lr, iterations, test_auc in roc_auc_values:
        model = CatBoostClassifier(depth=depth, learning_rate=lr, iterations=iterations, cat_features=cat_features)
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.plot(fpr, tpr, lw=2, label=f"Depth: {depth}, LR: {lr}, Iter: {iterations}\n(Val AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=config.TITLE_FONTSIZE)
    plt.ylabel("True Positive Rate", fontsize=config.TITLE_FONTSIZE)
    plt.title("ROC Curves for Top 5 Parameter Combinations on Internal Validation Set", fontsize=20)
    plt.legend(loc="lower right", fontsize="large")
    plt.show()

# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    """Main workflow: data loading -> preprocessing -> model training -> evaluation -> visualization -> hyperparameter search"""
    # Initialize configuration
    config = ModelConfig()

    # Step 1: Load all data
    print("="*50 + " Loading Data " + "="*50)
    data_dict = load_all_data(config)

    # Step 2: Preprocess PFS data
    print("\n" + "="*50 + " Preprocessing PFS Data " + "="*50)
    before_data_PFS = data_dict["before_PFS"]
    X_final, X, y, outside_validation = preprocess_pfs_data(before_data_PFS, config)

    # Step 3: Standardize features
    X = standardize_features(X, config)

    # Step 4: Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.CATBOOST_PARAMS["random_seed"]
    )
    print(f"\nTrain set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Step 5: Train CatBoost model
    print("\n" + "="*50 + " Training CatBoost Model " + "="*50)
    catboost_model, feature_importance, train_pool = train_catboost_model(X_train, X_test, y_train, y_test, config)

    # Step 6: Evaluate model
    print("\n" + "="*50 + " Model Evaluation " + "="*50)
    metrics, train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc = evaluate_catboost_model(
        catboost_model, X_train, X_test, y_train, y_test
    )

    # Step 7: Visualization
    print("\n" + "="*50 + " Visualization " + "="*50)
    plot_roc_curve(train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc, config)
    plot_learning_curve(catboost_model, config)
    plot_km_survival(data_dict["before_OS"], before_data_PFS, config)
    plot_top_features(feature_importance, train_pool, config)

    # Step 8: Hyperparameter search
    print("\n" + "="*50 + " Hyperparameter Search " + "="*50)
    catboost_parameter_search(X_train, y_train, X_test, y_test, config)

if __name__ == "__main__":
    main()
