import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from typing import Tuple, List, Dict, Optional

# Configuration class: Manages all configurable parameters for easy unified modification
class PlotConfig:
    """Configuration class for DCA and Calibration Curve plotting parameters"""
    # Plot style configuration for DCA curves (train/test/external validation)
    DCA_STYLES = {
        'train': {'color': 'deepskyblue', 'label': 'train model', 'all_label': 'treat all'},
        'test': {'color': 'orange', 'label': 'Internal Validation', 'all_label': 'test all'},
        'external': {'color': 'green', 'label': 'validation', 'all_label': 'validation treat all'}
    }
    # Plot style configuration for Calibration curves (train/test/external validation)
    CALIBRATION_STYLES = {
        'train': {'color': 'red', 'label': 'Model'},
        'test': {'color': 'deepskyblue', 'label': 'Model_test'},
        'external': {'color': 'green', 'label': 'Model_validation'}
    }
    # Font and axis label configuration (Times New Roman for academic plotting)
    FONT_CONFIG = {'family': 'Times New Roman', 'fontsize': 20}
    # Threshold range for DCA (start, end, step) - probability thresholds from 0 to 1
    THRESH_RANGE = (0, 1, 0.01)
    # Calibration curve parameters: number of bins and binning strategy
    CALIBRATION_BINS = 4
    CALIBRATION_STRATEGY = "quantile"  # Alternative: "uniform" for equal-width bins

# Data processing utility functions
def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV dataset and extract true labels and prediction scores, with missing value handling.
    The CSV file must follow the format: [Column 0: true labels, Column 1: prediction scores].
    
    Args:
        file_path: Absolute/relative path to the CSV data file.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - y_label: Numpy array of true binary labels (0/1) with NaNs removed
            - y_pred_score: Numpy array of model prediction scores (probabilities) with NaNs removed
            
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the CSV file has fewer than 2 columns (insufficient data).
    """
    try:
        # Read CSV file and convert to numpy array for numerical processing
        df = pd.read_csv(file_path)
        if df.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns: [true_labels, prediction_scores]")
        
        npdf = df.to_numpy()
        y_pred_score = npdf[:, 1]  # Prediction scores in second column
        y_label = npdf[:, 0]       # True labels in first column

        # Remove rows with NaN values (critical for valid metric calculation)
        mask = ~np.isnan(y_pred_score) & ~np.isnan(y_label)
        y_pred_score = y_pred_score[mask]
        y_label = y_label[mask]

        return y_label, y_pred_score
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at the specified path: {file_path}")

def calculate_error(y_pred_score: np.ndarray) -> float:
    """
    Calculate the standard deviation of prediction scores as the error metric for calibration curves.
    The standard deviation quantifies the spread of predicted probabilities, used for error bars.
    
    Args:
        y_pred_score: Numpy array of model prediction scores (probabilities).
        
    Returns:
        float: Standard deviation of the prediction scores (error value).
    """
    return np.std(y_pred_score)

# DCA (Decision Curve Analysis) calculation functions
def calculate_net_benefit_model(thresh_group: np.ndarray, y_pred_score: np.ndarray, y_label: np.ndarray) -> np.ndarray:
    """
    Calculate the net benefit of a predictive model across a range of probability thresholds.
    Net benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold)), where N = total samples.
    
    Args:
        thresh_group: Numpy array of probability thresholds (0 to 1).
        y_pred_score: Model prediction scores (probabilities) for the dataset.
        y_label: True binary labels (0/1) for the dataset.
        
    Returns:
        np.ndarray: Net benefit values for the model at each threshold.
    """
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        # Convert continuous scores to binary predictions using current threshold
        y_pred_label = y_pred_score > thresh
        # Compute confusion matrix components (TN, FP, FN, TP)
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)  # Total number of samples
        
        # Avoid division by zero when threshold = 1 (1 - thresh = 0)
        if (1 - thresh) == 0:
            net_benefit = 0.0
        else:
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group: np.ndarray, y_label: np.ndarray) -> np.ndarray:
    """
    Calculate the net benefit of the "treat all" strategy (predict all samples as positive)
    across a range of probability thresholds. This serves as a baseline for DCA comparison.
    
    Args:
        thresh_group: Numpy array of probability thresholds (0 to 1).
        y_label: True binary labels (0/1) for the dataset.
        
    Returns:
        np.ndarray: Net benefit values for the "treat all" strategy at each threshold.
    """
    net_benefit_all = np.array([])
    tp_total = np.sum(y_label == 1)  # Total true positives in dataset
    total = len(y_label)             # Total samples
    fp_total = total - tp_total      # All negative samples are false positives in "treat all"
    
    for thresh in thresh_group:
        # Avoid division by zero when threshold = 1
        if (1 - thresh) == 0:
            net_benefit = 0.0
        else:
            net_benefit = (tp_total / total) - (fp_total / total) * (thresh / (1 - thresh))
        
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def combine_datasets_and_calculate_net_benefit_model(
    thresh_group: np.ndarray,
    y_pred_score_dataset1: np.ndarray, y_label_dataset1: np.ndarray,
    y_pred_score_dataset2: np.ndarray, y_label_dataset2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine two datasets and calculate the combined net benefit of the model.
    Useful for merging training and validation data for aggregated DCA analysis.
    
    Args:
        thresh_group: Numpy array of probability thresholds (0 to 1).
        y_pred_score_dataset1/y_label_dataset1: Scores and labels for the first dataset.
        y_pred_score_dataset2/y_label_dataset2: Scores and labels for the second dataset.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - net_benefit_model_combined: Combined net benefit values for the model.
            - y_label_combined: Combined true labels from both datasets.
    """
    # Concatenate the two datasets along the sample axis
    y_pred_score_combined = np.concatenate((y_pred_score_dataset1, y_pred_score_dataset2))
    y_label_combined = np.concatenate((y_label_dataset1, y_label_dataset2))
    
    # Calculate net benefit for the combined dataset
    net_benefit_model_combined = calculate_net_benefit_model(thresh_group, y_pred_score_combined, y_label_combined)
    return net_benefit_model_combined, y_label_combined

# Core plotting functions for DCA and Calibration curves
def plot_dca_curve(
    ax: plt.Axes,
    thresh_group: np.ndarray,
    net_benefit_model: np.ndarray,
    net_benefit_all: np.ndarray,
    plot_type: str = 'train'
) -> plt.Axes:
    """
    Plot a single DCA curve (train/test/external) on a given Matplotlib Axes object.
    Integrates the model's net benefit and the "treat all" baseline, with style customization.
    
    Args:
        ax: Matplotlib Axes object to draw the DCA curve on.
        thresh_group: Numpy array of probability thresholds (0 to 1).
        net_benefit_model: Net benefit values for the predictive model.
        net_benefit_all: Net benefit values for the "treat all" baseline strategy.
        plot_type: Type of curve to plot ('train'/'test'/'external') - maps to PlotConfig.DCA_STYLES.
        
    Returns:
        plt.Axes: Updated Matplotlib Axes object with the DCA curve plotted.
    """
    # Retrieve style configuration for the specified plot type
    style = PlotConfig.DCA_STYLES[plot_type]
    
    # Plot model net benefit curve and "treat all" baseline curve
    ax.plot(thresh_group, net_benefit_model, color=style['color'], label=style['label'])
    ax.plot(
        thresh_group, net_benefit_all,
        color='slategrey' if plot_type != 'train' else 'black',
        label=style['all_label']
    )
    
    # Plot "treat none" baseline (only once for train to avoid duplicate lines)
    if plot_type == 'train':
        ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

    # Configure axis limits, labels, and visual style (academic publication standards)
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.2)
    ax.set_xlabel('Threshold Probability', fontdict=PlotConfig.FONT_CONFIG)
    ax.set_ylabel('Net Benefit', fontdict=PlotConfig.FONT_CONFIG)
    ax.grid(False)  # Disable grid for cleaner academic plots
    ax.spines['right'].set_color((0.8, 0.8, 0.8))  # Light gray for right spine
    ax.spines['top'].set_color((0.8, 0.8, 0.8))    # Light gray for top spine
    ax.legend(loc='lower left')  # Position legend to avoid overlapping curves

    return ax

def plot_calibration_curve(
    ax: plt.Axes,
    y_label: np.ndarray,
    y_pred_score: np.ndarray,
    error: float,
    plot_type: str = 'train'
) -> plt.Axes:
    """
    Plot a calibration curve with error bars on a given Matplotlib Axes object.
    Calibration curves assess how well model predictions match actual probabilities,
    with error bars showing the standard deviation of prediction scores.
    
    Args:
        ax: Matplotlib Axes object to draw the calibration curve on.
        y_label: True binary labels (0/1) for the dataset.
        y_pred_score: Model prediction scores (probabilities) for the dataset.
        error: Standard deviation of prediction scores (for error bars).
        plot_type: Type of curve to plot ('train'/'test'/'external') - maps to PlotConfig.CALIBRATION_STYLES.
        
    Returns:
        plt.Axes: Updated Matplotlib Axes object with the calibration curve plotted.
    """
    # Retrieve style configuration for the specified plot type
    style = PlotConfig.CALIBRATION_STYLES[plot_type]
    
    # Calculate calibration curve metrics (fraction of positives vs. predicted values)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_label, y_pred_score,
        pos_label=1,  # Positive class label (1 for binary classification)
        n_bins=PlotConfig.CALIBRATION_BINS,
        strategy=PlotConfig.CALIBRATION_STRATEGY
    )

    # Plot error bars (standard deviation) and scatter points for calibration bins
    ax.errorbar(
        mean_predicted_value, fraction_of_positives,
        yerr=error, color=style['color'], label=style['label'],
        capsize=5, capthick=2, linewidth=1.5  # Error bar styling
    )
    ax.scatter(
        mean_predicted_value, fraction_of_positives,
        facecolors='none', edgecolor=style['color'], s=10, zorder=5
    )
    # Plot perfect calibration line (45-degree line) for reference
    ax.plot((0, 1), (0, 1), color='black', linestyle=':')

    # Configure axis limits, labels, and visual style
    ax.set_xlim(0, 1)  # Prediction scores range from 0 to 1
    ax.set_xlabel('Predicted Value', fontdict=PlotConfig.FONT_CONFIG)
    ax.set_ylabel('Fraction of Positives', fontdict=PlotConfig.FONT_CONFIG)
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper left')  # Position legend to avoid overlapping the 45-degree line

    return ax

# Main execution function (entry point of the script)
def main(
    train_path: str,
    test_path: str,
    external_path: str
) -> None:
    """
    Main function to orchestrate data loading, metric calculation, and plotting of DCA/calibration curves.
    This function is the core workflow: load data → compute metrics → plot combined curves for train/test/external.
    
    Args:
        train_path: Path to the training dataset CSV file.
        test_path: Path to the internal validation (test) dataset CSV file.
        external_path: Path to the external validation dataset CSV file.
    """
    # Step 1: Load datasets and calculate error metrics (standard deviation of prediction scores)
    y_label_train, y_pred_score_train = load_data(train_path)
    y_label_test, y_pred_score_test = load_data(test_path)
    y_label_external, y_pred_score_external = load_data(external_path)

    error_train = calculate_error(y_pred_score_train)
    error_test = calculate_error(y_pred_score_test)
    error_validation = calculate_error(y_pred_score_external)

    # Print error metrics for debugging/analysis
    print(f"Training set prediction score standard deviation: {error_train:.4f}")
    print(f"Internal test set prediction score standard deviation: {error_test:.4f}")
    print(f"External validation set prediction score standard deviation: {error_validation:.4f}")

    # Step 2: Generate probability threshold group for DCA (0 to 1 with 0.01 step)
    thresh_group = np.arange(*PlotConfig.THRESH_RANGE)

    # Step 3: Calculate DCA metrics for all datasets (train/test/external)
    dca_metrics = {
        'train': (calculate_net_benefit_model(thresh_group, y_pred_score_train, y_label_train),
                  calculate_net_benefit_all(thresh_group, y_label_train)),
        'test': (calculate_net_benefit_model(thresh_group, y_pred_score_test, y_label_test),
                 calculate_net_benefit_all(thresh_group, y_label_test)),
        'external': (calculate_net_benefit_model(thresh_group, y_pred_score_external, y_label_external),
                     calculate_net_benefit_all(thresh_group, y_label_external))
    }

    # Step 4: Plot combined DCA curve (train + test + external validation)
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size for academic plots
    for plot_type in ['train', 'test', 'external']:
        net_benefit_model, net_benefit_all = dca_metrics[plot_type]
        ax = plot_dca_curve(ax, thresh_group, net_benefit_model, net_benefit_all, plot_type)
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()

    # Step 5: Plot combined Calibration curve (train + test + external validation)
    fig_1, ax_1 = plt.subplots(figsize=(10, 6))
    ax_1 = plot_calibration_curve(ax_1, y_label_train, y_pred_score_train, error_train, 'train')
    ax_1 = plot_calibration_curve(ax_1, y_label_test, y_pred_score_test, error_test, 'test')
    ax_1 = plot_calibration_curve(ax_1, y_label_external, y_pred_score_external, error_validation, 'external')
    plt.tight_layout()
    plt.show()

# Script entry point (execute only when run directly, not imported as a module)
if __name__ == '__main__':
    # Replace the following paths with your actual data file paths (use raw string r"" to avoid escape errors)
    # Example: r"C:/Users/YourName/AD/DCA/DCA OS.csv"
    TRAIN_PATH = r"your path/DCA OS.csv"
    TEST_PATH = r"your path/DCA OS test.csv"
    EXTERNAL_PATH = r"your path/DCA validation.csv"

    # Run the main workflow
    main(TRAIN_PATH, TEST_PATH, EXTERNAL_PATH)
