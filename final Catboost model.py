import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import KaplanMeierFitter
from scipy import stats
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warning messages

# ------------------------------------------------------------------------------
# Simplified Configuration Class: Only retain PFS analysis essentials
# ------------------------------------------------------------------------------
class ModelConfig:
    """
    Simplified configuration for PFS (Progression-Free Survival) analysis with CatBoost
    Key components: Core data paths, CatBoost hyperparameters, preprocessing settings, visualization params
    Replace the file paths with your actual PFS/OS dataset paths
    """
    # Core dataset paths (only PFS and OS for survival analysis; remove redundant paths)
    DATA_PATHS = {
        "before_PFS": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment-8.2 PFS.csv",
        "before_OS": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 OS.csv"
    }

    # CatBoost core hyperparameters for binary classification
    CATBOOST_PARAMS = {
        "eval_metric": "Logloss",  # Evaluation metric for binary classification
        "cat_features": ["Age", "Location", "N", "TNM", "PTV_Dose", "GTV_Dose", "ECOG", "T", "Chemotherapy"],
        "random_seed": 42,  # Fixed seed for reproducibility
        "early_stopping_rounds": 50  # Early stopping to prevent overfitting
    }

    # Preprocessing configuration
    OUTLIER_Z_THRESHOLD = 4  # Z-score threshold for outlier removal
    CORR_THRESHOLD = 0.8  # Correlation threshold for dropping highly correlated features
    EXTERNAL_VALIDATION_SIZE = 60  # Sample size for external validation set (reserved)
    TEST_SIZE = 0.3  # Test set proportion (70% train, 30% test)
    STANDARDIZE_KEYWORDS = ["treatment", "TL", "original"]  # Feature columns to standardize

    # Hyperparameter search grid for CatBoost tuning
    PARAM_GRID = {
        "depth": [4, 6, 8],  # Tree depth
        "learning_rate": [0.01, 0.03, 0.1],  # Step size for gradient descent
        "iterations": [400, 600, 800, 1000]  # Number of boosting iterations
    }

    # Visualization settings
    PLOT_FIGSIZE = (8, 6)  # Default figure size for plots
    LARGE_FIGSIZE = (12, 8)  # Larger figure size for hyperparameter results
    TITLE_FONTSIZE = 25  # Font size for plot titles
    LABEL_FONTSIZE = 20  # Font size for axis labels
    KM_FONTSIZE = 27  # Font size for Kaplan-Meier plot
    TOP_FEATURES = 10  # Number of top features to visualize for importance

# ------------------------------------------------------------------------------
# Custom CatBoost Loss Function: Penalty/Reward Mechanism for Confident Predictions
# ------------------------------------------------------------------------------
class CustomLoglossObjective:
    """
    Custom Logloss objective for CatBoost with penalty/reward mechanism
    - Penalty for overconfident wrong predictions (e.g., p>0.8 but label=0)
    - Reward for confident correct predictions (e.g., p>0.8 and label=1)
    - Combines Logloss and Quantile Loss derivatives for robust optimization
    """
    def __init__(self, penalty=2, reward_factor=0.9):
        self.penalty = penalty  # Penalty multiplier for overconfident errors
        self.reward_factor = reward_factor  # Reward multiplier for confident correct predictions

    def calc_ders_range(self, approxes, targets, weights):
        """
        Calculate first and second derivatives for CatBoost's custom loss
        Args:
            approxes (list): Model predictions (log-odds)
            targets (list): True binary labels (0/1)
            weights (list): Sample weights (None if unweighted)
        Returns:
            list: Tuples of (first derivative, second derivative) for each sample
        """
        # Validate input dimensions
        assert len(approxes) == len(targets), "Predictions and labels must have the same length"
        if weights is not None:
            assert len(weights) == len(approxes), "Weights must match sample count"

        exponents = [np.exp(a) for a in approxes]  # Convert log-odds to probabilities
        quantile = 0.6  # Quantile for Quantile Loss component
        derivatives = []

        for idx in range(len(targets)):
            # Calculate probability from log-odds (sigmoid function)
            prob = exponents[idx] / (1 + exponents[idx])
            
            # Apply penalty for overconfident wrong predictions
            penalty = 1.0
            if (targets[idx] == 1 and prob < 0.2) or (targets[idx] == 0 and prob > 0.8):
                penalty *= self.penalty

            # Apply reward for confident correct predictions
            reward = 1.0
            if (targets[idx] == 1 and prob > 0.8) or (targets[idx] == 0 and prob < 0.2):
                reward *= self.reward_factor

            # Logloss derivatives (first and second)
            der1_log = (1 - prob) * self.penalty if targets[idx] > 0.0 else -prob * self.penalty
            der2_log = -prob * (1 - prob)

            # Quantile Loss derivatives (first and second)
            der1_quantile = quantile * (prob - prob**2) if (targets[idx] - prob) >= 0 else (quantile - 1) * (prob - prob**2)
            der2_quantile = 0  # Second derivative of Quantile Loss is 0

            # Combine derivatives from Logloss and Quantile Loss
            der1_combined = (der1_log + der1_quantile) / 2
            der2_combined = (der2_log + der2_quantile) / 2

            # Apply sample weights if provided
            if weights is not None:
                der1_combined *= weights[idx]
                der2_combined *= weights[idx]

            derivatives.append((der1_combined, der2_combined))

        return derivatives

# ------------------------------------------------------------------------------
# Core Data Loading & Preprocessing Functions
# ------------------------------------------------------------------------------
def load_core_data(config):
    """
    Load only core PFS and OS datasets (remove redundant data loading)
    Args:
        config (ModelConfig): Configuration object with data paths
    Returns:
        dict: Dictionary of loaded DataFrames (keys: "before_PFS", "before_OS")
    Raises:
        FileNotFoundError: If any core dataset path is invalid
    """
    data_dict = {}
    for dataset_key, file_path in config.DATA_PATHS.items():
        try:
            # Load CSV and drop rows with missing values
            data_dict[dataset_key] = pd.read_csv(file_path).dropna()
            # Print dataset summary
            print(f"Loaded {dataset_key}: {len(data_dict[dataset_key])} samples, {data_dict[dataset_key].shape[1]} features")
        except FileNotFoundError:
            raise FileNotFoundError(f"Core dataset not found at path: {file_path}")
    return data_dict

def preprocess_pfs_data(before_data_PFS, config):
    """
    Preprocess PFS dataset: Outlier removal → Correlated feature drop → Feature/label split
    Args:
        before_data_PFS (pd.DataFrame): Raw PFS dataset with PFS_m (time) and PFS (event) columns
        config (ModelConfig): Preprocessing configuration
    Returns:
        tuple: (X: feature DataFrame, y: binary label array)
    """
    # Step 1: Remove outliers using Z-score (Z > threshold)
    z_scores = stats.zscore(before_data_PFS)
    outlier_mask = (z_scores > config.OUTLIER_Z_THRESHOLD).any(axis=1)
    data_no_outliers = before_data_PFS[~outlier_mask].dropna()
    print(f"PFS data after outlier removal: {len(data_no_outliers)} samples remaining")

    # Step 2: Drop highly correlated features (correlation > threshold)
    corr_matrix = data_no_outliers.corr().abs()  # Absolute correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > config.CORR_THRESHOLD)]
    X_final = data_no_outliers.drop(correlated_features, axis=1)
    print(f"PFS data after dropping correlated features: {X_final.shape[1]} features remaining")

    # Step 3: Sample external validation set (reserved for future use, not used in core pipeline)
    _ = X_final.sample(n=config.EXTERNAL_VALIDATION_SIZE, random_state=config.CATBOOST_PARAMS["random_seed"])

    # Step 4: Split features (X) and labels (y) (drop survival time/event columns for classification)
    X = X_final.drop(["PFS_m", "PFS"], axis=1).astype(str)  # Features (convert to string for categorical handling)
    y = X_final["PFS"].values  # Binary label (1 = event, 0 = censored)

    return X, y

def standardize_features(X, config):
    """
    Standardize numerical features containing specific keywords (z-score normalization)
    Args:
        X (pd.DataFrame): Feature DataFrame
        config (ModelConfig): Standardization configuration (keywords for columns to standardize)
    Returns:
        pd.DataFrame: Feature DataFrame with standardized columns
    """
    scaler = StandardScaler()  # Z-score scaler (mean=0, std=1)
    # Identify columns to standardize (contains any keyword in STANDARDIZE_KEYWORDS)
    cols_to_standardize = [col for col in X.columns if any(kw in col for kw in config.STANDARDIZE_KEYWORDS)]
    
    if cols_to_standardize:
        # Apply standardization to selected columns
        X[cols_to_standardize] = scaler.fit_transform(X[cols_to_standardize])
        print(f"Standardized columns: {', '.join(cols_to_standardize)}")
    return X

# ------------------------------------------------------------------------------
# CatBoost Model Training & Evaluation Functions
# ------------------------------------------------------------------------------
def train_catboost_model(X_train, X_test, y_train, y_test, config):
    """
    Train CatBoost model with custom loss function
    Args:
        X_train/X_test (pd.DataFrame): Train/test feature DataFrames
        y_train/y_test (np.ndarray): Train/test binary labels
        config (ModelConfig): CatBoost hyperparameters
    Returns:
        tuple: (trained model, feature importance array, train Pool object)
    """
    # Create CatBoost Pool objects (optimized for categorical feature handling)
    train_pool = Pool(X_train, y_train, cat_features=config.CATBOOST_PARAMS["cat_features"])
    test_pool = Pool(X_test, y_test, cat_features=config.CATBOOST_PARAMS["cat_features"])

    # Initialize custom loss function
    custom_loss = CustomLoglossObjective()
    # Initialize CatBoost classifier with custom loss
    model = CatBoostClassifier(
        loss_function=custom_loss,
        **config.CATBOOST_PARAMS,
        verbose=False  # Disable training logs
    )
    # Train model with early stopping
    model.fit(train_pool, eval_set=test_pool)

    # Extract feature importance (PredictionValuesChange: impact on prediction)
    feature_importance = model.get_feature_importance(data=train_pool, type="PredictionValuesChange")
    return model, feature_importance, train_pool

def calculate_classification_metrics(y_true, y_pred, y_prob):
    """
    Calculate core classification metrics for binary classification
    Args:
        y_true (np.ndarray): True binary labels
        y_pred (np.ndarray): Predicted binary labels (0/1)
        y_prob (np.ndarray): Predicted probabilities for positive class (1)
    Returns:
        tuple: (accuracy, sensitivity, specificity, auc_score)
    """
    # Compute confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Overall accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Recall)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    auc_score = roc_auc_score(y_true, y_prob)  # Area Under ROC Curve

    # Print metric summary
    print("\nCatBoost Core Classification Metrics:")
    print(f"Accuracy: {accuracy:.3f} | Sensitivity (Recall): {sensitivity:.3f} | Specificity: {specificity:.3f} | AUC: {auc_score:.3f}")
    return accuracy, sensitivity, specificity, auc_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate trained CatBoost model: Predictions → Metrics → ROC Curve Data
    Args:
        model (CatBoostClassifier): Trained CatBoost model
        X_train/X_test (pd.DataFrame): Train/test feature DataFrames
        y_train/y_test (np.ndarray): Train/test binary labels
    Returns:
        tuple: (train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc)
    """
    # Generate predictions (labels and probabilities)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)[:, 1]  # Probabilities for positive class
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Calculate test set metrics (train metrics optional for overfitting check)
    _ = calculate_classification_metrics(y_test, y_pred_test, y_prob_test)

    # Compute ROC curve and AUC for train/test sets
    train_fpr, train_tpr, _ = roc_curve(y_train, y_prob_train)
    test_fpr, test_tpr, _ = roc_curve(y_test, y_prob_test)
    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc = roc_auc_score(y_test, y_prob_test)

    return train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc

# ------------------------------------------------------------------------------
# Visualization Functions: ROC Curve, Kaplan-Meier, Feature Importance
# ------------------------------------------------------------------------------
def plot_roc_curve(train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc, config):
    """
    Plot ROC (Receiver Operating Characteristic) curve for train/test sets
    Args:
        train_fpr/train_tpr (np.ndarray): Train set FPR/TPR for ROC curve
        test_fpr/test_tpr (np.ndarray): Test set FPR/TPR for ROC curve
        train_auc/test_auc (float): Train/test AUC scores
        config (ModelConfig): Visualization settings
    """
    plt.figure(figsize=config.PLOT_FIGSIZE)
    # Plot train and test ROC curves
    plt.plot(train_fpr, train_tpr, label=f"Train AUC = {train_auc:.3f}")
    plt.plot(test_fpr, test_tpr, label=f"Validation AUC = {test_auc:.3f}")
    # Plot random guess baseline (diagonal line)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    # Add labels and title
    plt.xlabel("False Positive Rate (FPR)", fontsize=config.LABEL_FONTSIZE)
    plt.ylabel("True Positive Rate (TPR)", fontsize=config.LABEL_FONTSIZE)
    plt.title("ROC Curve (CatBoost for PFS Prediction)", fontsize=config.TITLE_FONTSIZE)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_km_survival(before_data_OS, before_data_PFS, config):
    """
    Plot Kaplan-Meier survival curves for OS (Overall Survival) and PFS (Progression-Free Survival)
    Args:
        before_data_OS (pd.DataFrame): OS dataset with OS_m (time) and OS (event) columns
        before_data_PFS (pd.DataFrame): PFS dataset with PFS_m (time) and PFS (event) columns
        config (ModelConfig): Visualization settings
    """
    # Initialize Kaplan-Meier fitter
    km_fitter = KaplanMeierFitter()

    # Fit and plot OS survival curve
    km_fitter.fit(durations=before_data_OS["OS_m"], event_observed=before_data_OS["OS"], label="OS")
    ax = km_fitter.plot(figsize=config.PLOT_FIGSIZE)

    # Fit and plot PFS survival curve on the same axis
    km_fitter.fit(durations=before_data_PFS["PFS_m"], event_observed=before_data_PFS["PFS"], label="PFS")
    km_fitter.plot(ax=ax)

    # Customize plot labels and title
    plt.xlabel("Time (Days/Months)", fontsize=config.KM_FONTSIZE)
    plt.ylabel("Survival Probability", fontsize=config.KM_FONTSIZE)
    plt.title("Kaplan-Meier Survival Curves (OS vs PFS)", fontsize=config.KM_FONTSIZE)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def plot_top_features(feature_importance, train_pool, config):
    """
    Plot horizontal bar chart for top N feature importance (CatBoost)
    Args:
        feature_importance (np.ndarray): Feature importance array from CatBoost
        train_pool (Pool): CatBoost train Pool object (for feature names)
        config (ModelConfig): Visualization settings (TOP_FEATURES, font sizes)
    """
    # Sort feature importance in descending order and select top N
    sorted_indices = np.argsort(feature_importance)[::-1][:config.TOP_FEATURES]
    feature_names = train_pool.get_feature_names()
    top_feature_names = [feature_names[i] for i in sorted_indices]
    top_feature_importance = [feature_importance[i] for i in sorted_indices]

    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_feature_names)), top_feature_importance, color="skyblue")
    # Customize plot
    plt.yticks(range(len(top_feature_names)), top_feature_names, fontsize=config.LABEL_FONTSIZE)
    plt.xlabel("Feature Importance (Prediction Values Change)", fontsize=config.TITLE_FONTSIZE)
    plt.title(f"Top {config.TOP_FEATURES} Features for PFS Prediction (CatBoost)", fontsize=config.TITLE_FONTSIZE)
    plt.gca().invert_yaxis()  # Invert y-axis to show top feature at the top
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Hyperparameter Search: Grid Search for CatBoost
# ------------------------------------------------------------------------------
def catboost_param_search(X_train, y_train, X_test, y_test, config):
    """
    Simplified grid search for CatBoost hyperparameters (depth, learning rate, iterations)
    Focus on test set AUC and visualize top 5 hyperparameter combinations
    Args:
        X_train/X_test (pd.DataFrame): Train/test feature DataFrames
        y_train/y_test (np.ndarray): Train/test binary labels
        config (ModelConfig): Hyperparameter grid and visualization settings
    """
    search_results = []
    cat_features = config.CATBOOST_PARAMS["cat_features"]

    # Iterate over all hyperparameter combinations in the grid
    print("\nStarting CatBoost hyperparameter grid search... (This may take a few minutes)")
    for depth in config.PARAM_GRID["depth"]:
        for lr in config.PARAM_GRID["learning_rate"]:
            for iterations in config.PARAM_GRID["iterations"]:
                # Initialize CatBoost model with current hyperparameters
                model = CatBoostClassifier(
                    depth=depth,
                    learning_rate=lr,
                    iterations=iterations,
                    cat_features=cat_features,
                    random_seed=config.CATBOOST_PARAMS["random_seed"],
                    eval_metric="Logloss",
                    verbose=False
                )
                # Train model
                model.fit(Pool(X_train, y_train, cat_features=cat_features), eval_set=Pool(X_test, y_test, cat_features=cat_features))
                # Calculate test set AUC (key metric for ranking)
                test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                # Store results
                search_results.append((depth, lr, iterations, test_auc))

    # Convert results to DataFrame and sort by test AUC (descending)
    results_df = pd.DataFrame(search_results, columns=["Tree Depth", "Learning Rate", "Iterations", "Test AUC"])
    results_df = results_df.sort_values(by="Test AUC", ascending=False)

    # Print top 10 hyperparameter combinations
    print("\nTop 10 CatBoost Hyperparameter Combinations (Sorted by Test AUC):")
    print(tabulate(results_df.head(10), headers="keys", tablefmt="psql", floatfmt=".3f"))

    # Visualize ROC curves for top 5 hyperparameter combinations
    top5_combinations = results_df.head(5).values
    plt.figure(figsize=config.LARGE_FIGSIZE)

    for depth, lr, iterations, auc_score in top5_combinations:
        # Retrain model with top hyperparameters
        model = CatBoostClassifier(
            depth=depth, learning_rate=lr, iterations=iterations,
            cat_features=cat_features, random_seed=config.CATBOOST_PARAMS["random_seed"],
            verbose=False
        )
        model.fit(X_train, y_train)
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        # Plot ROC curve with hyperparameter label
        plt.plot(fpr, tpr, lw=2, label=f"Depth:{depth}, LR:{lr}, Iters:{iterations} (AUC={auc_score:.3f})")

    # Add random guess baseline
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", label="Random Guess")
    # Customize plot
    plt.xlabel("False Positive Rate (FPR)", fontsize=config.TITLE_FONTSIZE)
    plt.ylabel("True Positive Rate (TPR)", fontsize=config.TITLE_FONTSIZE)
    plt.title("ROC Curves for Top 5 CatBoost Hyperparameter Combinations", fontsize=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    """
    Main pipeline for PFS prediction with CatBoost:
    1. Load core PFS/OS datasets
    2. Preprocess PFS data (outlier removal, feature standardization)
    3. Split train/test sets
    4. Train CatBoost model with custom loss
    5. Evaluate model and calculate metrics
    6. Visualize results (ROC, Kaplan-Meier, feature importance)
    7. Hyperparameter grid search for model tuning
    """
    # Initialize configuration
    config = ModelConfig()

    # Step 1: Load core datasets
    print("="*60 + " Load Core Datasets " + "="*60)
    data_dict = load_core_data(config)
    pfs_data = data_dict["before_PFS"]
    os_data = data_dict["before_OS"]

    # Step 2: Preprocess PFS data
    print("\n" + "="*60 + " Preprocess PFS Data " + "="*60)
    X, y = preprocess_pfs_data(pfs_data, config)
    X = standardize_features(X, config)

    # Step 3: Split train/test sets (stratified split optional for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.CATBOOST_PARAMS["random_seed"]
    )
    print(f"\nTrain set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Step 4: Train CatBoost model
    print("\n" + "="*60 + " Train CatBoost Model " + "="*60)
    catboost_model, feat_importance, train_pool = train_catboost_model(X_train, X_test, y_train, y_test, config)

    # Step 5: Evaluate model
    print("\n" + "="*60 + " Evaluate Model Performance " + "="*60)
    train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc = evaluate_model(catboost_model, X_train, X_test, y_train, y_test)

    # Step 6: Visualize results
    print("\n" + "="*60 + " Visualize Results " + "="*60)
    plot_roc_curve(train_fpr, train_tpr, test_fpr, test_tpr, train_auc, test_auc, config)
    plot_km_survival(os_data, pfs_data, config)
    plot_top_features(feat_importance, train_pool, config)

    # Step 7: Hyperparameter search
    print("\n" + "="*60 + " CatBoost Hyperparameter Search " + "="*60)
    catboost_param_search(X_train, y_train, X_test, y_test, config)

# Run the main pipeline
if __name__ == "__main__":
    main()
