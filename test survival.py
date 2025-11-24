import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFECV, RFE
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import Lasso
import itertools
from sklearn.svm import SVC
from forestplot import forestplot
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# ------------------------------------------------------------------------------
# Configuration Class: Manage core paths and hyperparameters
# ------------------------------------------------------------------------------
class CoxConfig:
    """
    Configuration for Cox Proportional Hazards Model analysis (PFS-focused)
    Only retains core datasets used in the analysis (removes redundant OS/LC/LRFS data)
    """
    # Core data paths (only PFS-related and essential datasets)
    DATA_PATHS = {
        "new_PFS_train": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.2 PFS.csv",
        "new_PFS_test": r"D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 PFS external validation.csv",
        "new_data": r"D:\Acsq\BIM硕士第三学期\AD/before OS 7.19.csv",
        "final_data": r"D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 OS after preprocessing.csv"
    }

    # Lasso feature selection hyperparameters
    LASSO_ALPHA = 1.0

    # Cox model settings
    DURATION_COL = "PFS_m"
    EVENT_COL = "PFS"
    TOP_FEATURES = 10  # Number of top features to plot
    RANDOM_STATE = 42  # Fixed random seed for reproducibility

    # Visualization settings
    PLOT_FIGSIZE = (12, 8)
    SMALL_FIGSIZE = (10, 6)
    TITLE_FONTSIZE = 27
    LABEL_FONTSIZE = 27
    TICK_FONTSIZE = 25

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------------------------
def load_core_data(config):
    """
    Load only core PFS-related datasets and drop missing values
    Args:
        config (CoxConfig): Configuration object with data paths
    Returns:
        dict: Dictionary of loaded DataFrames (PFS train/test, new_data)
    Raises:
        FileNotFoundError: If any core data file path is invalid
    """
    data_dict = {}
    for key, path in config.DATA_PATHS.items():
        try:
            data_dict[key] = pd.read_csv(path).dropna()
            print(f"Loaded {key} dataset with {len(data_dict[key])} samples and {data_dict[key].shape[1]} features")
        except FileNotFoundError:
            raise FileNotFoundError(f"Core data file not found: {path}")
    
    # Validate essential columns exist in PFS datasets
    for pfs_key in ["new_PFS_train", "new_PFS_test"]:
        if pfs_key in data_dict:
            assert config.DURATION_COL in data_dict[pfs_key].columns, f"{config.DURATION_COL} missing in {pfs_key}"
            assert config.EVENT_COL in data_dict[pfs_key].columns, f"{config.EVENT_COL} missing in {pfs_key}"
    
    return data_dict

def lasso_feature_selection(new_data, config):
    """
    Perform Lasso regression for feature selection (OS-focused)
    Args:
        new_data (pd.DataFrame): Raw dataset with OS/OS_m columns
        config (CoxConfig): Lasso hyperparameters
    Returns:
        pd.DataFrame: Dataset with selected features + OS + OS_m
    """
    # Split features and labels (OS prediction)
    if "OS" not in new_data.columns or "OS_m" not in new_data.columns:
        raise ValueError("new_data must contain 'OS' and 'OS_m' columns for Lasso selection")
    
    X = new_data.drop(['OS', 'OS_m'], axis=1)
    y = new_data['OS']
    time = new_data['OS_m']

    # Train Lasso model
    lasso = Lasso(alpha=config.LASSO_ALPHA, random_state=config.RANDOM_STATE)
    lasso.fit(X, y)

    # Select non-zero coefficient features
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    print(f"\nLasso selected {len(selected_features)} features: {selected_features}")

    # Create new dataset with selected features
    selected_data = X[selected_features]
    new_data_after_lasso = pd.concat([selected_data, y, time], axis=1)
    
    return new_data_after_lasso

# ------------------------------------------------------------------------------
# Cox Model Training and Evaluation Functions
# ------------------------------------------------------------------------------
def train_evaluate_cox_model(X_train, X_test, config):
    """
    Train Cox Proportional Hazards Model and evaluate with Concordance Index
    Plots top feature importance and Cox model survival curve
    Args:
        X_train (pd.DataFrame): Training dataset (contains duration/event columns)
        X_test (pd.DataFrame): Test dataset (contains duration/event columns)
        config (CoxConfig): Cox model settings and visualization params
    Returns:
        tuple: (cph model, test c-index, top features)
    """
    # Initialize and train Cox model
    cph = CoxPHFitter()
    cph.fit(X_train, duration_col=config.DURATION_COL, event_col=config.EVENT_COL)

    # Predict and calculate Concordance Index on test set
    test_predictions = cph.predict_expectation(X_test)
    test_c_index = concordance_index(
        X_test[config.DURATION_COL], 
        test_predictions, 
        X_test[config.EVENT_COL]
    )
    print(f"\nCox Model Test Concordance Index: {test_c_index:.3f}")

    # Print model summary
    print("\nCox Model Training Summary:")
    cph.print_summary()

    # Plot top N feature importance (by absolute coefficient)
    feature_importances = cph.params_
    top_features = feature_importances.abs().sort_values(ascending=False).head(config.TOP_FEATURES)
    
    plt.figure(figsize=config.PLOT_FIGSIZE)
    top_features.plot(kind='barh')
    plt.title(f'Top {config.TOP_FEATURES} Features by Importance', fontsize=config.TITLE_FONTSIZE)
    plt.xlabel('Coefficient Value', fontsize=config.LABEL_FONTSIZE)
    plt.ylabel('Features', fontsize=config.LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.gca().tick_params(axis='y', labelsize=config.TICK_FONTSIZE)
    plt.tight_layout()
    plt.show()

    # Plot Cox model survival curve
    plt.figure(figsize=config.SMALL_FIGSIZE)
    cph.plot()
    plt.title('Cox Proportional Hazards Model Survival Curve', fontsize=20)
    plt.xlabel('Log-Hazard Ratio', fontsize=18)
    plt.ylabel('Survival Function', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['Cox Model'], loc='best')
    plt.tight_layout()
    plt.show()

    return cph, test_c_index, top_features.index.tolist()

def optimize_feature_combination(pfs_train, pfs_test, config):
    """
    Optimize feature combination for Cox model by random feature selection
    Finds the best feature set with maximum Concordance Index
    Args:
        pfs_train (pd.DataFrame): PFS training dataset
        pfs_test (pd.DataFrame): PFS test dataset
        config (CoxConfig): Cox model settings
    Returns:
        tuple: (best_c_index, best_features, c_index_values)
    """
    # Split essential columns and feature columns
    essential_cols = [config.DURATION_COL, config.EVENT_COL]
    feature_cols = [col for col in pfs_train.columns if col not in essential_cols]
    total_features = len(feature_cols)
    print(f"\nStarting feature combination optimization (total features: {total_features})")

    # Initialize tracking variables
    c_index_values = []
    max_c_index = 0.0
    best_features = essential_cols

    # Iterate over different feature counts (1 to total features)
    for n_features in range(1, total_features + 1):
        # Randomly select n features (fixed random seed for reproducibility)
        np.random.seed(config.RANDOM_STATE)
        selected_features = np.random.choice(feature_cols, n_features, replace=False).tolist()
        current_cols = essential_cols + selected_features

        # Train Cox model on selected features
        cph = CoxPHFitter()
        cph.fit(pfs_train[current_cols], duration_col=config.DURATION_COL, event_col=config.EVENT_COL)

        # Calculate train C-index (using train set for faster iteration)
        train_predictions = cph.predict_expectation(pfs_train[current_cols])
        train_c_index = concordance_index(
            pfs_train[config.DURATION_COL],
            train_predictions,
            pfs_train[config.EVENT_COL]
        )
        c_index_values.append(train_c_index)

        # Update best feature set if current C-index is higher
        if train_c_index > max_c_index:
            max_c_index = train_c_index
            best_features = current_cols

        print(f"Number of features: {n_features} | Train C-index: {train_c_index:.3f}")

    # Evaluate best feature set on test set
    cph_best = CoxPHFitter()
    cph_best.fit(pfs_train[best_features], duration_col=config.DURATION_COL, event_col=config.EVENT_COL)
    test_predictions_best = cph_best.predict_expectation(pfs_test[best_features])
    test_c_index_best = concordance_index(
        pfs_test[config.DURATION_COL],
        test_predictions_best,
        pfs_test[config.EVENT_COL]
    )

    # Print optimization results
    print(f"\n=== Feature Optimization Results ===")
    print(f"Best train C-index: {max_c_index:.3f} (features: {len(best_features)-2})")
    print(f"Best feature combination: {[f for f in best_features if f not in essential_cols]}")
    print(f"Test C-index with best features: {test_c_index_best:.3f}")
    print(f"\nBest Cox Model Summary:")
    cph_best.print_summary()

    # Plot C-index vs number of features
    plt.figure(figsize=config.SMALL_FIGSIZE)
    plt.plot(range(1, total_features + 1), c_index_values, marker='o', color='blue', linestyle='-')
    plt.title('Concordance Index vs Number of Features', fontsize=20)
    plt.xlabel('Number of Features', fontsize=18)
    plt.ylabel('Train Concordance Index', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot survival curve for best feature set
    plt.figure(figsize=config.SMALL_FIGSIZE)
    cph_best.plot()
    plt.title('Cox Model Survival Curve (Best Features)', fontsize=20)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Survival Probability', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return test_c_index_best, best_features, c_index_values

# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    """Main workflow for Cox PH Model analysis (PFS-focused)"""
    # Initialize configuration
    config = CoxConfig()
    print("="*60 + " Cox Proportional Hazards Model Analysis " + "="*60)

    # Step 1: Load core data (only PFS-related datasets)
    print("\n" + "="*50 + " Loading Core Data " + "="*50)
    data_dict = load_core_data(config)
    new_PFS_train = data_dict["new_PFS_train"]
    new_PFS_test = data_dict["new_PFS_test"]
    new_data = data_dict["new_data"]

    # Step 2: Lasso feature selection (OS-focused)
    print("\n" + "="*50 + " Lasso Feature Selection " + "="*50)
    new_data_after_lasso = lasso_feature_selection(new_data, config)
    print(f"Lasso-processed dataset shape: {new_data_after_lasso.shape}")

    # Step 3: Train and evaluate baseline Cox model (PFS)
    print("\n" + "="*50 + " Baseline Cox Model Training " + "="*50)
    cph_baseline, test_c_index, top_features = train_evaluate_cox_model(new_PFS_train, new_PFS_test, config)

    # Step 4: Optimize feature combination for Cox model
    print("\n" + "="*50 + " Feature Combination Optimization " + "="*50)
    best_test_c_index, best_features, c_index_values = optimize_feature_combination(new_PFS_train, new_PFS_test, config)

    # Final results summary
    print("\n" + "="*60 + " Final Analysis Results " + "="*60)
    print(f"Baseline Cox Model Test C-index: {test_c_index:.3f}")
    print(f"Optimized Cox Model Test C-index: {best_test_c_index:.3f}")
    print(f"Top features from baseline model: {top_features}")
    print(f"Best feature combination: {[f for f in best_features if f not in [config.DURATION_COL, config.EVENT_COL]]}")

if __name__ == "__main__":
    main()
