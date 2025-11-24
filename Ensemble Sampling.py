import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tabulate import tabulate
import seaborn as sns

# Configuration class: Manages all hyperparameters and file paths
class ModelConfig:
    """Configuration for model training, data paths and visualization"""
    # File paths (replace with your actual paths)
    DATA_PATHS = {
        'train': r"your path/EC_before_Treatment labeled 8.2 PFS.csv",
        'test': r"your path/EC_before_Treatment test data 8.1 PFS.csv",
        'validation': r"your path/EC_before_Treatment 7.12 PFS external validation.csv",
        'full': r"your path/EC_before_Treatment 7.12 OS after preprocessing.csv",
        'drop': r"your path/EC_before_Treatment_unlabeled_samples 22 7.22.csv"
    }
    # CatBoost hyperparameters
    CATBOOST_PARAMS = {
        'depth': 8,
        'iterations': 400,
        'learning_rate': 0.1,
        'early_stopping_rounds': 50,
        'eval_metric': 'Logloss',
        'random_seed': 42
    }
    # Categorical features for CatBoost
    CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose', 'ECOG', 'T', 'Chemotherapy']
    # Features to standardize (keywords matching)
    STANDARDIZE_KEYWORDS = ['treatment', 'TL', 'original']
    # Label noise levels
    NOISE_LEVELS = {
        'train': 0.05,
        'test': 0.01,
        'validation': 0.01
    }
    # Visualization settings
    PLOT_FIGSIZE = (8, 6)
    PLOT_TITLE_FONTSIZE = 25
    PLOT_LABEL_FONTSIZE = 20
    TOP_FEATURES = 10  # Number of top features to show in importance plot

# Custom Logloss Objective for CatBoost
class CustomLoglossObjective:
    """
    Custom Logloss Objective for CatBoost with penalty/reward mechanism
    Combines Logloss and Quantile Loss with dynamic penalty for overconfident wrong predictions
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
        Calculate first and second derivatives for CatBoost custom loss
        Args:
            approxes: Model predictions (list of floats)
            targets: True labels (list of floats)
            weights: Sample weights (list of floats, optional)
        Returns:
            list: Pairs of (first derivative, second derivative)
        """
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = [np.exp(a) for a in approxes]
        gamma = 2
        beta = 0.5
        result = []

        for idx in range(len(targets)):
            p = exponents[idx] / (1 + exponents[idx])
            penalty = 1.0

            # Apply penalty for overconfident wrong predictions
            if (targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8):
                penalty *= self.penalty

            # Apply reward for confident correct predictions
            reward_factor = 1.0
            if targets[idx] == 1 and p > 0.8:
                reward_factor *= self.reward_factor
            elif targets[idx] == 0 and p < 0.2:
                reward_factor *= self.reward_factor

            # Logloss derivatives
            der1_log = (1 - p) * self.penalty if targets[idx] > 0.0 else -p * self.penalty
            der2_log = -p * (1 - p)

            # Quantile loss derivatives (q=0.6)
            q = 0.6
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

# Data processing functions
def load_data(file_paths: dict) -> tuple:
    """
    Load and preprocess all datasets (drop NA values)
    Args:
        file_paths: Dictionary of data file paths
    Returns:
        tuple: Train, test, validation, full, drop datasets (pd.DataFrame)
    Raises:
        FileNotFoundError: If any file path is invalid
    """
    try:
        train_data = pd.read_csv(file_paths['train']).dropna()
        test_data = pd.read_csv(file_paths['test']).dropna()
        validation = pd.read_csv(file_paths['validation']).dropna()
        full_data = pd.read_csv(file_paths['full']).dropna()
        drop_data = pd.read_csv(file_paths['drop']).dropna()
        return train_data, test_data, validation, full_data, drop_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e.filename}")

def add_label_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add random noise to binary labels by flipping a percentage of samples
    Args:
        y: Original binary labels (0/1)
        noise_level: Fraction of labels to flip (0 to 1)
    Returns:
        np.ndarray: Noisy labels
    """
    num_samples = len(y)
    num_noise = int(noise_level * num_samples)
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    
    noisy_y = np.copy(y)
    noisy_y[noise_indices] = 1 - noisy_y[noise_indices]
    return noisy_y

def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame, 
                         keywords: list) -> tuple:
    """
    Standardize selected features (matching keywords) using StandardScaler
    Args:
        X_train/X_test/X_val: Feature DataFrames
        keywords: List of keywords to identify features for standardization
    Returns:
        tuple: Standardized X_train, X_test, X_val
    """
    scaler = StandardScaler()
    columns_to_standardize = [col for col in X_train.columns if any(kw in col for kw in keywords)]
    
    # Fit scaler on training data only to avoid data leakage
    X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
    X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])
    X_val[columns_to_standardize] = scaler.transform(X_val[columns_to_standardize])
    
    return X_train, X_test, X_val

def prepare_features_labels(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                            val_data: pd.DataFrame, noise_levels: dict) -> tuple:
    """
    Prepare features (X) and labels (y) for training, test and validation sets
    Includes label noise addition and feature standardization
    Args:
        train_data/test_data/val_data: Raw DataFrames
        noise_levels: Dictionary of noise levels for each dataset
    Returns:
        tuple: X_train, y_train, y_train_noisy, X_test, y_test, y_test_noisy, X_val, y_val, y_val_noisy
    """
    # Extract features and labels (drop PFS/PFS_m)
    X_train = train_data.drop(['PFS', 'PFS_m'], axis=1).astype(str)
    y_train = train_data['PFS'].values
    X_test = test_data.drop(['PFS', 'PFS_m'], axis=1).astype(str)
    y_test = test_data['PFS'].values
    X_val = val_data.drop(['PFS', 'PFS_m'], axis=1).astype(str)
    y_val = val_data['PFS'].values

    # Add noise to labels
    y_train_noisy = add_label_noise(y_train, noise_levels['train'])
    y_test_noisy = add_label_noise(y_test, noise_levels['test'])
    y_val_noisy = add_label_noise(y_val, noise_levels['validation'])

    # Standardize selected features
    X_train, X_test, X_val = standardize_features(
        X_train, X_test, X_val, ModelConfig.STANDARDIZE_KEYWORDS
    )

    return X_train, y_train, y_train_noisy, X_test, y_test, y_test_noisy, X_val, y_val, y_val_noisy

# Model training functions
def train_catboost(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
                   cat_features: list, params: dict) -> CatBoostClassifier:
    """
    Train CatBoost classifier with custom loss function
    Args:
        X_train/y_train: Training data
        X_test/y_test: Validation data for early stopping
        cat_features: List of categorical feature names
        params: CatBoost hyperparameters
    Returns:
        CatBoostClassifier: Trained model
    """
    # Create CatBoost Pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    # Initialize model with custom loss
    model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=cat_features,
        **params
    )

    # Train model
    model.fit(train_pool, eval_set=test_pool, verbose=False)
    return model

def train_baseline_models(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """
    Train baseline classification models (RandomForest, LogisticRegression, KNN, SVM, AdaBoost)
    Args:
        X_train/y_train: Training data
    Returns:
        dict: Trained baseline models
    """
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }

    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name} model")

    return models

# Metric calculation functions
def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate key classification metrics: Accuracy, Sensitivity, Specificity, Precision, F1-Score
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        tuple: (accuracy, sensitivity, specificity, precision, f1_score)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return accuracy, sensitivity, specificity, precision, f1_score

def compute_metrics_summary(train_metrics: tuple, test_metrics: tuple, val_metrics: tuple) -> pd.DataFrame:
    """
    Compute summary metrics (mean ± std) for train, test and validation sets
    Args:
        train_metrics/test_metrics/val_metrics: Metrics tuples from calculate_classification_metrics
    Returns:
        pd.DataFrame: Summary of individual and aggregated metrics
    """
    # Individual metrics DataFrame
    individual_metrics = pd.DataFrame({
        'Dataset': ['Train', 'Test', 'Validation'],
        'Accuracy': [train_metrics[0], test_metrics[0], val_metrics[0]],
        'Sensitivity': [train_metrics[1], test_metrics[1], val_metrics[1]],
        'Specificity': [train_metrics[2], test_metrics[2], val_metrics[2]],
        'Precision': [train_metrics[3], test_metrics[3], val_metrics[3]],
        'F1 Score': [train_metrics[4], test_metrics[4], val_metrics[4]]
    })

    # Calculate mean and std
    all_metrics = np.array([train_metrics, test_metrics, val_metrics])
    mean_vals = np.mean(all_metrics, axis=0)
    std_vals = np.std(all_metrics, axis=0, ddof=1)

    # Format mean ± std
    metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']
    mean_std = pd.Series(
        [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_vals, std_vals)],
        index=metrics_names
    )

    return individual_metrics, mean_std

def calculate_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> tuple:
    """
    Calculate ROC curve points and AUC score
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (positive class)
    Returns:
        tuple: (fpr, tpr, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    return fpr, tpr, auc_score

# Visualization functions
def plot_roc_curve(roc_data: dict, title: str, figsize: tuple, label_fontsize: int, title_fontsize: int):
    """
    Plot ROC curve for multiple datasets/models
    Args:
        roc_data: Dictionary of {label: (fpr, tpr, auc)}
        title: Plot title
        figsize: Figure size (width, height)
        label_fontsize: Font size for axis labels
        title_fontsize: Font size for plot title
    """
    plt.figure(figsize=figsize)
    for label, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{label} AUC = {auc_score:.3f}')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    
    # Format plot
    plt.xlabel('False Positive Rate', fontsize=label_fontsize)
    plt.ylabel('True Positive Rate', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend()
    plt.show()

def plot_catboost_learning_curve(model: CatBoostClassifier, figsize: tuple, label_fontsize: int, title_fontsize: int):
    """
    Plot CatBoost learning curve (train vs validation loss)
    Args:
        model: Trained CatBoost model
        figsize: Figure size
        label_fontsize: Axis label font size
        title_fontsize: Title font size
    """
    evals_result = model.get_evals_result()
    train_loss = evals_result['learn']['Logloss']
    val_loss = evals_result['validation']['Logloss']

    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.xlabel('Number of Iterations', fontsize=label_fontsize)
    plt.ylabel('Logloss', fontsize=label_fontsize)
    plt.title('CatBoost Learning Curve', fontsize=title_fontsize)
    plt.legend()
    plt.show()

def plot_top_features(model: CatBoostClassifier, pool: Pool, top_n: int, figsize: tuple = (10, 6)):
    """
    Plot top N feature importance from CatBoost model
    Args:
        model: Trained CatBoost model
        pool: CatBoost Pool object (training data)
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Get feature importance
    feature_importance = model.get_feature_importance(data=pool, type='PredictionValuesChange')
    feature_names = pool.get_feature_names()

    # Sort and select top N features
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_indices = sorted_indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]

    # Plot horizontal bar chart
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importance, color='skyblue', align='center')
    plt.yticks(range(len(top_features)), top_features, fontsize=25)
    plt.xlabel('Importance', fontsize=25)
    plt.title(f'Top {top_n} Important Features', fontsize=30)
    plt.gca().invert_yaxis()  # Invert y-axis to show top feature at the top
    plt.show()

# Main execution function
def main():
    """Main workflow for data processing, model training, evaluation and visualization"""
    # 1. Load configuration and data
    config = ModelConfig()
    train_data, test_data, val_data, full_data, drop_data = load_data(config.DATA_PATHS)

    # 2. Prepare features and labels
    X_train, y_train, y_train_noisy, X_test, y_test, y_test_noisy, X_val, y_val, y_val_noisy = prepare_features_labels(
        train_data, test_data, val_data, config.NOISE_LEVELS
    )

    # 3. Train CatBoost model
    print("Training CatBoost model...")
    catboost_model = train_catboost(
        X_train, y_train, X_test, y_test,
        config.CAT_FEATURES, config.CATBOOST_PARAMS
    )

    # 4. Make predictions with CatBoost
    cb_train_preds = catboost_model.predict(X_train)
    cb_test_preds = catboost_model.predict(X_test)
    cb_val_preds = catboost_model.predict(X_val)

    cb_train_probs = catboost_model.predict_proba(X_train)[:, 1]
    cb_test_probs = catboost_model.predict_proba(X_test)[:, 1]
    cb_val_probs = catboost_model.predict_proba(X_val)[:, 1]

    # 5. Calculate classification metrics for CatBoost
    train_metrics = calculate_classification_metrics(y_train_noisy, cb_train_preds)
    test_metrics = calculate_classification_metrics(y_test_noisy, cb_test_preds)
    val_metrics = calculate_classification_metrics(y_val_noisy, cb_val_preds)

    # 6. Print metrics summary
    individual_metrics, mean_std_metrics = compute_metrics_summary(train_metrics, test_metrics, val_metrics)
    print("\nIndividual Dataset Metrics:")
    print(individual_metrics)
    print("\nOverall Metrics (Mean ± Std):")
    print(mean_std_metrics)

    # 7. Calculate ROC-AUC for CatBoost (train/test/validation)
    cb_roc_train = calculate_roc_auc(y_train_noisy, cb_train_probs)
    cb_roc_test = calculate_roc_auc(y_test, cb_test_probs)
    cb_roc_val = calculate_roc_auc(y_val, cb_val_probs)

    # Plot CatBoost ROC curve (all datasets)
    roc_data_cb = {
        'Train': cb_roc_train,
        'Internal Validation': cb_roc_test,
        'External Validation': cb_roc_val
    }
    plot_roc_curve(
        roc_data_cb,
        'ROC Curve (CatBoost)',
        config.PLOT_FIGSIZE,
        config.PLOT_LABEL_FONTSIZE,
        config.PLOT_TITLE_FONTSIZE
    )

    # 8. Plot CatBoost learning curve and feature importance
    train_pool = Pool(X_train, y_train, cat_features=config.CAT_FEATURES)
    plot_catboost_learning_curve(
        catboost_model,
        config.PLOT_FIGSIZE,
        config.PLOT_LABEL_FONTSIZE,
        config.PLOT_TITLE_FONTSIZE
    )
    plot_top_features(catboost_model, train_pool, config.TOP_FEATURES)

    # 9. Train baseline models
    print("\nTraining baseline models...")
    baseline_models = train_baseline_models(X_train, y_train)

    # 10. Calculate ROC-AUC for baseline models (train and test)
    # Train set ROC data
    roc_data_train = {'CatBoost': cb_roc_train}
    # Test set ROC data
    roc_data_test = {'CatBoost': cb_roc_test}

    for name, model in baseline_models.items():
        # Train set predictions
        train_probs = model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, auc_train = calculate_roc_auc(y_train, train_probs)
        roc_data_train[name] = (fpr_train, tpr_train, auc_train)

        # Test set predictions
        test_probs = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, auc_test = calculate_roc_auc(y_test, test_probs)
        roc_data_test[name] = (fpr_test, tpr_test, auc_test)

    # Plot baseline models ROC curves
    plot_roc_curve(
        roc_data_train,
        'Train ROC Curve Comparison',
        config.PLOT_FIGSIZE,
        config.PLOT_LABEL_FONTSIZE,
        config.PLOT_TITLE_FONTSIZE
    )

    plot_roc_curve(
        roc_data_test,
        'Internal Validation ROC Curve Comparison',
        config.PLOT_FIGSIZE,
        config.PLOT_LABEL_FONTSIZE,
        config.PLOT_TITLE_FONTSIZE
    )

if __name__ == '__main__':
    main()
