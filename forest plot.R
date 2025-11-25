# Load required libraries
library(forestplot)
library(ggplot2)
library(dplyr)
library(caret)
library(grid) # Required for gpar and unit functions

# ------------------------------------------------------------------------------
# Configuration: Centralized path and plot parameters 
# ------------------------------------------------------------------------------
# Data file paths (modify these paths according to your local environment)
FILE_PATHS <- list(
  forest_data = "your path/forest plot.csv",
  dca_train = "your path/DCA PFS.csv",
  dca_validation = "your path/DCA validation.csv"
)

# Forest plot visual parameters
FOREST_PARAMS <- list(
  boxsize = 0.2,          # Size of the HR box
  graph_pos = 2,          # Position of the forest plot graph
  graph_width = unit(.25, "npc"), # Width of the forest plot
  clip_range = c(0.1, 8), # X-axis range for clipping
  zero_line = 1,          # Reference line for HR=1
  ci_vertices_height = 0.2 # Height of CI vertices
)

# Calibration curve visual parameters
CALIBRATION_PARAMS <- list(
  base_size = 15,         # Base font size for ggplot
  line_color = "crimson", # Color of calibration curve
  line_size = 1           # Thickness of calibration curve
)

# ------------------------------------------------------------------------------
# Function 1: Plot Forest Plot for Hazard Ratio
# ------------------------------------------------------------------------------
#' Plot Hazard Ratio Forest Plot
#' 
#' This function loads forest plot data, preprocesses it, and generates a forest plot for hazard ratio (HR)
#' with customized visual styles and confidence interval settings.
#' 
#' @param data_path Character path to the CSV file containing forest plot data
#' @param plot_params List of visual parameters for forest plot (boxsize, graph_pos, etc.)
#' @return A forest plot object (printed automatically)
plot_hr_forest <- function(data_path, plot_params) {
  # Step 1: Load and preprocess data
  data <- read.csv(data_path, stringsAsFactors = FALSE)
  
  # Create CI range column (combine lower and upper 95% CI)
  data$CI_range <- paste(data$exp.coef..lower.95., data$exp.coef..upper95., sep = "-")
  
  # Correct the header of CI_range column
  data[1, 14] <- "CI_range"
  
  # Convert numeric columns
  data[, c(3, 7, 8)] <- lapply(data[, c(3, 7, 8)], as.numeric)
  
  # Convert character columns
  data[, c(12, 10, 14)] <- lapply(data[, c(12, 10, 14)], as.character)
  
  # Step 2: Generate forest plot
  fig <- forestplot(
    # Select display columns
    data[, c(1, 12, 10, 14)],
    # HR value and 95% CI
    mean = data[, 3],
    lower = data[, 7],
    upper = data[, 8],
    # Reference line
    zero = plot_params$zero_line,
    # Box size
    boxsize = plot_params$boxsize,
    # Graph position
    graph.pos = plot_params$graph_pos,
    # Horizontal lines style
    hrzl_lines = list(
      "1" = gpar(lty = 1, lwd = 2),  # Mean line
      "2" = gpar(lty = 2),            # Dashed line between CI
      "36" = gpar(lwd = 2, lty = 1)   # CI boundary line
    ),
    # Graph width
    graphwidth = plot_params$graph_width,
    # Text style
    txt_gp = fpTxtGp(
      label = gpar(cex = 0.7, fontface = "bold"),  # Label font
      ticks = gpar(cex = 1),                      # Tick font
      xlab = gpar(cex = 0.9),                     # X-axis label font
      title = gpar(cex = 1.5, col = "darkred")    # Title font and color
    ),
    # Line widths
    lwd.zero = 1,        # Reference line width
    lwd.ci = 1.5,        # CI line width
    lwd.xaxis = 2,       # X-axis line width
    lty.ci = 1.5,        # CI line type
    # CI vertices 
    ci.vertices = TRUE,
    ci.vertices.height = plot_params$ci_vertices_height,
    # X-axis clip range
    clip = plot_params$clip_range,
    # Spacing settings 
    ineheight = unit(8, 'mm'),
    line.margin = unit(8, 'mm'),
    colgap = unit(6, 'mm'),
    # CI shape 
    fn.ci_norm = "fpDrawDiamondCI",
    # Plot title 
    title = "Hazard Ratio",
    # Color settings 
    col = fpColors(
      box = "blue4",    # HR box color
      lines = "blue4",  # CI line color
      zero = "black"    # Reference line color
    )
  )
  
  # Print the forest plot
  print(fig)
  return(fig) # Return plot object for further customization
}

# ------------------------------------------------------------------------------
# Function 2: Plot Calibration Curve
# ------------------------------------------------------------------------------
#' Plot Calibration Curve for Predicted Probabilities
#' 
#' This function generates calibration curves to evaluate the agreement between predicted probabilities
#' and actual observed fractions of positive outcomes, split into 10 probability bins.
#' 
#' @param data Data frame containing true labels and predicted probabilities
#' @param label_col Character name of the true label column
#' @param prob_col Character name of the predicted probability column
#' @param plot_params List of visual parameters for calibration curve (base_size, line_color, etc.)
#' @param title Character title for the calibration curve plot
#' @return A ggplot object of the calibration curve
plot_calibration <- function(data, label_col, prob_col, plot_params, title) {
  # Preprocess data: bin predicted probabilities into 10 intervals
  calib_data <- data %>%
    mutate(predicted_category = cut(!!sym(prob_col), 
                                    breaks = seq(0, 1, by = 0.1), 
                                    include.lowest = TRUE)) %>%
    group_by(predicted_category) %>%
    summarize(
      fraction_of_positives = mean(!!sym(label_col)), # Observed positive fraction
      mean_predicted_value = mean(!!sym(prob_col)),   # Mean predicted probability
      count = n() # Number of samples in each bin
    ) %>%
    ungroup() # Ungroup to avoid dplyr issues
  
  # Generate calibration curve
  p <- ggplot(calib_data, aes(x = mean_predicted_value, y = fraction_of_positives)) +
    # Calibration curve line
    geom_line(color = plot_params$line_color, size = plot_params$line_size) +
    # Perfect calibration diagonal line
    geom_abline(slope = 1, intercept = 0, linetype = 'dashed', color = 'black') +
    # Axis labels 
    labs(x = 'Predicted Probability', y = 'Fraction of Positives', title = title) +
    # Theme settings 
    theme_minimal(base_size = plot_params$base_size) +
    theme(
      axis.text = element_text(family = 'Times New Roman'),
      axis.title = element_text(family = 'Times New Roman'),
      plot.title = element_text(family = 'Times New Roman', hjust = 0.5) # Center title
    )
  
  return(p)
}

# ------------------------------------------------------------------------------
# Function 3: Main Execution 
# ------------------------------------------------------------------------------
#' Main Function to Run All Plots
#' 
#' This function coordinates the execution of forest plot and calibration curve generation,
#' loads data, and arranges plots in a unified layout.
run_analysis_plots <- function() {
  # Step 1: Plot Hazard Ratio Forest Plot 
  cat("Generating Hazard Ratio Forest Plot...\n")
  forest_plot <- plot_hr_forest(FILE_PATHS$forest_data, FOREST_PARAMS)
  
  # Step 2: Load DCA data 
  cat("Loading DCA training and validation data...\n")
  df_train <- read.csv(FILE_PATHS$dca_train)
  df_validation <- read.csv(FILE_PATHS$dca_validation)
  
  # Step 3: Generate Calibration Curves 
  cat("Generating Calibration Curves...\n")
  calib_train <- plot_calibration(
    data = df_train,
    label_col = 'true_label',
    prob_col = 'predicted_prob',
    plot_params = CALIBRATION_PARAMS,
    title = 'Calibration Curve - Training Data'
  )
  
  calib_validation <- plot_calibration(
    data = df_validation,
    label_col = 'true_label',
    prob_col = 'predicted_prob',
    plot_params = CALIBRATION_PARAMS,
    title = 'Calibration Curve - Validation Data'
  )
  
  # Step 4: Arrange Calibration Curves in 1x2 Layout 
  # Use gridExtra for multi-plot layout 
  if (!require("gridExtra")) {
    install.packages("gridExtra")
    library(gridExtra)
  }
  grid.arrange(calib_train, calib_validation, ncol = 2)
  
  # Return all plot objects 
  return(list(
    forest_plot = forest_plot,
    calibration_train = calib_train,
    calibration_validation = calib_validation
  ))
}

# ------------------------------------------------------------------------------
# Run the entire analysis 
# ------------------------------------------------------------------------------
all_plots <- run_analysis_plots()
