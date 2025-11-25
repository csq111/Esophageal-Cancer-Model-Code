# Load required libraries (install missing packages with install.packages())
library(survival)
library(survminer)
library(ggplot2)
library(dplyr)
library(readxl)
library(extrafont)

# ------------------------------------------------------------------------------
# Centralized Configuration: File Paths & Visual Parameters
# ------------------------------------------------------------------------------
# Update these paths to your local file system
FILE_PATHS <- list(
  # OS/PFS primary data
  os_data = "your path/ OS.csv",
  pfs_data = "your path/ PFS.csv",
  os_data_1 = "your path/ OS 1.csv",
  
  # PFS KM prediction data
  pfs_km = "D:/Acsq/BIM硕士第三学期/AD/DCA/PFS KM.csv",
  pfs_km_validation = "D:/Acsq/BIM硕士第三学期/AD/DCA/PFS km.csv",
  
  # DCA & similarity plot data
  os_dca_validation = "D:/Acsq/BIM硕士第三学期/AD/DCA/OS DCA validation.csv",
  dca_os = "D:/Acsq/BIM硕士第三学期/AD/DCA/DCA OS.csv",
  os_similarity = "D:/Acsq/BIM硕士第三学期/AD/DCA/OS similarity plot - EV.csv",
  os_km_bland = "D:/Acsq/BIM硕士第三学期/AD/DCA/OS KM.csv",
  
  # IPI & Dual Expression data
  ipi_data = "C:/Users/Lenovo/Desktop/zyy/km/统计结果/IPI.csv",
  duet_data = "C:/Users/Lenovo/Desktop/zyy/program(1)/program/双表达/duet.csv",
  ipi_xlsx = "C:/Users/Lenovo/Desktop/zyy/program(1)/program/ipi/ipi.xlsx"
)

# Visualization parameters (consistent styling across plots)
VIS_PARAMS <- list(
  lancet_colors = c("#D55E00", "#56B4E9"),  # Lancet journal color scheme
  lancet_colors_alt = c("#CC79A7", "#0072B2"),
  x_axis_breaks = seq(0, 70, 10),           # X-axis tick interval for KM curves
  x_axis_limit = c(0, 70),                  # X-axis range for KM curves
  base_font_size = 16,                      # Base font size for plots
  large_font_size = 20,                     # Large font size for clinical plots
  risk_table_font = 13                      # Font size for risk tables
)

# ------------------------------------------------------------------------------
# Function 1: Plot KM Curves for OS vs PFS
# ------------------------------------------------------------------------------
#' Plot Kaplan-Meier Curves for Overall Survival (OS) and Progression-Free Survival (PFS)
#' 
#' @param os_path Character path to OS CSV data
#' @param pfs_path Character path to PFS CSV data
#' @param vis_params List of visualization parameters (colors, breaks, limits)
#' @return ggsurvplot object for OS/PFS KM curves
plot_os_pfs_km <- function(os_path, pfs_path, vis_params) {
  # Load raw data
  before_data_OS <- read.csv(os_path)
  before_data_PFS <- read.csv(pfs_path)
  
  # Combine OS and PFS data with group labels
  combined_data <- rbind(
    data.frame(time = before_data_OS$OS_m, status = before_data_OS$OS, group = "OS"),
    data.frame(time = before_data_PFS$PFS_m, status = before_data_PFS$PFS, group = "PFS")
  )
  
  # Create survival object
  surv_object <- Surv(time = combined_data$time, event = combined_data$status)
  
  # Fit Kaplan-Meier model
  km_fit <- survfit(surv_object ~ group, data = combined_data)
  
  # Generate KM plot with risk table
  plot <- ggsurvplot(
    km_fit,
    data = combined_data,
    break.time.by = diff(vis_params$x_axis_breaks)[1],
    risk.table = TRUE,
    risk.table.title = "",
    xlab = "Time (Months)",
    ylab = "Survival Probability",
    title = "Kaplan-Meier Curves for OS and PFS",
    palette = vis_params$lancet_colors_alt,
    legend.title = "Group",
    legend.labs = c("OS", "PFS"),
    conf.int = TRUE,
    censor.shape = 3,
    censor.size = 2,
    xlim = vis_params$x_axis_limit,
    ggtheme = theme_minimal(base_size = vis_params$base_font_size) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        legend.position = "right",
        legend.title = element_blank(),
        legend.text = element_text(size = 14, color = "black"),
        axis.title.x = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 14, color = "black"),
        axis.text.y = element_text(size = 14),
        panel.grid.major = element_line(linewidth = 0.5),
        panel.grid.minor = element_blank()
      )
  )
  
  # Update x-axis breaks
  plot$plot <- plot$plot + scale_x_continuous(breaks = vis_params$x_axis_breaks)
  
  # Print and return plot
  print(plot)
  return(plot)
}

# ------------------------------------------------------------------------------
# Function 2: Plot KM Curves for Predicted vs Observed PFS
# ------------------------------------------------------------------------------
#' Plot Kaplan-Meier Curves for Observed vs Predicted Progression-Free Survival (PFS)
#' 
#' @param pfs_km_path Character path to PFS prediction CSV data
#' @param vis_params List of visualization parameters (colors, breaks, limits)
#' @param is_validation Logical: TRUE for validation data, FALSE for test data
#' @return ggsurvplot object for predicted/observed PFS KM curves
plot_pfs_pred_obs_km <- function(pfs_km_path, vis_params, is_validation = FALSE) {
  # Load data
  data <- read.csv(pfs_km_path)
  
  # Combine data based on test/validation flag
  if (is_validation) {
    combined_data <- rbind(
      data.frame(time = data$live.time.validation, status = data$y_validation, group = "Train True"),
      data.frame(time = data$prediction.time.validation, status = data$y_validation_prediction, group = "Train Prediction")
    )
    ylab_label <- "Progression Free Survival"
    title_label <- ""
  } else {
    combined_data <- rbind(
      data.frame(time = data$live.time.test, status = data$y.test, group = "EV True"),
      data.frame(time = data$prediction.time.test, status = data$y.test.predictions, group = "EV Prediction")
    )
    ylab_label <- "Progression Free Survival"
    title_label <- "Kaplan-Meier Curves for True and Prediction"
  }
  
  # Create survival object
  surv_object <- Surv(time = combined_data$time, event = combined_data$status)
  
  # Fit Kaplan-Meier model
  km_fit <- survfit(surv_object ~ group, data = combined_data)
  
  # Generate KM plot with log-rank p-value
  plot <- ggsurvplot(
    km_fit,
    data = combined_data,
    break.time.by = diff(vis_params$x_axis_breaks)[1],
    risk.table = TRUE,
    pval = TRUE,
    pval.method = TRUE,
    pval.method.text = "logrank p",
    risk.table.title = "",
    risk.table.fontsize = vis_params$risk_table_font,
    risk.table.text = element_text(family = "Times New Roman"),
    xlab = "Time(month)",
    ylab = ylab_label,
    title = title_label,
    palette = vis_params$lancet_colors,
    legend.title = "",
    legend.labs = c("Observed", "Predicted"),
    conf.int = TRUE,
    censor.shape = 3,
    censor.size = 2,
    xlim = vis_params$x_axis_limit,
    ggtheme = theme_minimal(base_size = vis_params$large_font_size) +
      theme(
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(hjust = 1, size = 25, face = "bold"),
        plot.subtitle = element_text(size = 40, face = "bold"),
        legend.position = c(1, 1),
        legend.justification = c("right", "top"),
        legend.title = element_blank(),
        legend.text = element_text(size = 40, face = "bold", color = "black"),
        axis.title.x = element_text(size = 40, face = "bold", color = "black"),
        axis.title.y = element_text(size = 40, face = "bold"),
        axis.text.x = element_text(size = 35, face = "bold", color = "black"),
        axis.text.y = element_text(size = 35, face = "bold"),
        panel.grid.major = element_line(size = 0.7),
        panel.grid.minor = element_blank()
      ),
    pval.size = 10,
    pval.text = element_text(family = "Times New Roman")
  )
  
  # Update x-axis breaks
  plot$plot <- plot$plot + scale_x_continuous(breaks = vis_params$x_axis_breaks)
  
  # Print plot and summary
  print(plot)
  km_summary <- summary(km_fit)
  print(km_summary)
  
  return(list(plot = plot, summary = km_summary))
}

# ------------------------------------------------------------------------------
# Function 3: Plot Similarity Metrics (Confusion Matrix, Scatter, Bland-Altman)
# ------------------------------------------------------------------------------
#' Plot Similarity Metrics for Prediction Validation
#' 
#' Generates confusion matrix, correlation scatter plot, Bland-Altman plot, and error distribution
#' @param similarity_path Character path to similarity plot CSV data
#' @param bland_altman_path Character path to Bland-Altman CSV data
#' @param confusion_data Data frame for confusion matrix (optional)
#' @return List of ggplot objects for all similarity plots
plot_similarity_metrics <- function(similarity_path, bland_altman_path, confusion_data = NULL) {
  plots <- list()
  
  # 1. Confusion Matrix (if data provided)
  if (!is.null(confusion_data)) {
    confusion_matrix <- table(confusion_data$y.validation, confusion_data$validation.predictions)
    confusion_df <- as.data.frame(confusion_matrix)
    colnames(confusion_df) <- c("Actual", "Predicted", "Frequency")
    
    plots$confusion <- ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
      geom_tile() +
      geom_text(aes(label = Frequency), color = "white") +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
      theme_minimal()
    
    print(plots$confusion)
  }
  
  # 2. Correlation Scatter Plot (Predicted vs Observed Time)
  plotdata <- read.csv(similarity_path)
  test <- plotdata$live.time.validation
  prediction <- plotdata$prediction.time.validation
  label <- plotdata$y.validation
  correlation <- cor(test, prediction)
  
  plots$scatter <- ggplot(plotdata, aes(x = test, y = prediction)) +
    geom_point(aes(color = as.factor(label)), size = 1) +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    labs(title = "Prediction Time vs Live Time", x = "Prediction Time", y = "Live Time") +
    scale_color_manual(values = c("red", "blue"), name = "") +
    theme_minimal() +
    theme(
      text = element_text(family = "Times New Roman", size = 15),
      plot.title = element_text(hjust = 0.5, size = 15),
      legend.spacing.y = unit(0.5, "cm"),
      legend.position = c(0.8, 0.1),
      plot.background = element_rect(fill = "white")
    ) +
    xlim(0, 80) +
    ylim(0, 80) +
    annotate("text", x = 10, y = 70,
             label = paste("Correlation: ", round(correlation, 2)),
             size = 5, color = "black", family = "Times New Roman")
  
  print(plots$scatter)
  
  # 3. Bland-Altman Plot
  data_ba <- read.csv(bland_altman_path)
  data_ba$mean_train <- rowMeans(data_ba[, c('prediction.time.train', 'live.time.train')])
  data_ba$difference_train <- data_ba$prediction.time.train - data_ba$live.time.train
  mean_train <- mean(data_ba$difference_train, na.rm = TRUE)
  sd_train <- sd(data_ba$difference_train, na.rm = TRUE)
  
  plots$bland_altman <- ggplot(data_ba, aes(x = mean_train, y = difference_train)) +
    geom_point(size = 1, shape = 1, color = 'orangered') +
    geom_hline(yintercept = mean_train, color = "blue", linetype = "dashed") +
    geom_hline(yintercept = mean_train + 1.96 * sd_train, color = "red", linetype = "dashed") +
    geom_hline(yintercept = mean_train - 1.96 * sd_train, color = "red", linetype = "dashed") +
    labs(title = "Bland-Altman Plot", x = "Mean of Measurements", y = "Difference between Measurements") +
    theme_minimal()
  
  print(plots$bland_altman)
  
  # 4. Error Distribution Histogram
  error <- data_ba$live.time.train - data_ba$prediction.time.train
  plots$error_hist <- ggplot(data.frame(error), aes(x = error)) +
    geom_histogram(binwidth = 0.5, color = 'blue') +
    labs(title = "Error Distribution", x = "Error (True - Predicted)", y = "Frequency") +
    theme_minimal()
  
  print(plots$error_hist)
  
  return(plots)
}

# ------------------------------------------------------------------------------
# Function 4: Plot KM Curves for Risk Stratification (High/Low Risk)
# ------------------------------------------------------------------------------
#' Plot Kaplan-Meier Curves for High/Low Risk Stratification
#' 
#' @param dca_validation_path Character path to DCA validation CSV data
#' @param dca_os_path Character path to DCA OS training CSV data
#' @param vis_params List of visualization parameters (colors, breaks, limits)
#' @return ggsurvplot object for risk-stratified KM curves
plot_risk_strat_km <- function(dca_validation_path, dca_os_path, vis_params) {
  # Load data
  data <- read.csv(dca_validation_path)
  train_data <- read.csv(dca_os_path)
  
  # Define high/low risk groups based on median prediction
  median_pred <- median(train_data$y.train.predictions)
  data$risk_group <- ifelse(data$validation.predictions >= median_pred, "High Risk", "Low Risk")
  
  # Print median values for reference
  cat("Median training prediction:", median_pred, "\n")
  cat("Median test prediction:", median(data$y.test.predictions, na.rm = TRUE), "\n")
  
  # Create survival object
  surv_object <- Surv(time = data$live.time, event = data$y.validation)
  
  # Fit Kaplan-Meier model
  km_fit <- survfit(surv_object ~ risk_group, data = data)
  
  # Log-rank test (chi-square)
  chi_square_result <- survdiff(surv_object ~ risk_group, data = data)
  chi_square <- chi_square_result$chisq
  cat("Log-rank test chi-square:", chi_square, "\n")
  
  # Generate risk-stratified KM plot
  plot <- ggsurvplot(
    km_fit,
    data = data,
    break.time.by = diff(vis_params$x_axis_breaks)[1],
    risk.table = TRUE,
    pval = TRUE,
    pval.method = TRUE,
    pval.method.text = "logrank p",
    risk.table.title = "",
    risk.table.fontsize = vis_params$risk_table_font,
    risk.table.text = element_text(family = "Times New Roman"),
    xlab = "Time(month)",
    ylab = "Progression Free Survival",
    title = "",
    palette = vis_params$lancet_colors,
    legend.title = "",
    legend.labs = c("High Risk", "Low Risk"),
    conf.int = TRUE,
    censor.shape = 3,
    censor.size = 2,
    xlim = vis_params$x_axis_limit,
    ggtheme = theme_minimal(base_size = vis_params$large_font_size) +
      theme(
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(hjust = 1, size = 25, face = "bold"),
        plot.subtitle = element_text(size = 40, face = "bold"),
        legend.position = c(1, 1),
        legend.justification = c("right", "top"),
        legend.title = element_blank(),
        legend.text = element_text(size = 40, face = "bold", color = "black"),
        axis.title.x = element_text(size = 40, face = "bold", color = "black"),
        axis.title.y = element_text(size = 40, face = "bold"),
        axis.text.x = element_text(size = 35, face = "bold", color = "black"),
        axis.text.y = element_text(size = 35, face = "bold"),
        panel.grid.major = element_line(size = 0.7),
        panel.grid.minor = element_blank()
      ),
    pval.size = 10,
    pval.text = element_text(family = "Times New Roman")
  )
  
  # Update x-axis breaks
  plot$plot <- plot$plot + scale_x_continuous(breaks = vis_params$x_axis_breaks)
  
  # Print plot and summary
  print(plot)
  km_summary <- summary(km_fit)
  print(km_summary)
  
  return(list(plot = plot, summary = km_summary, chi_square = chi_square))
}

# ------------------------------------------------------------------------------
# Function 5: Plot KM Curves for IPI and Dual Expression (MYC)
# ------------------------------------------------------------------------------
#' Plot Kaplan-Meier Curves for IPI and Dual Expression (MYC) Stratification
#' 
#' @param ipi_path Character path to IPI CSV data
#' @param duet_path Character path to dual expression CSV data
#' @param ipi_xlsx_path Character path to IPI XLSX data
#' @param vis_params List of visualization parameters (colors, breaks, limits)
#' @return List of ggsurvplot objects for IPI and dual expression KM curves
plot_ipi_duet_km <- function(ipi_path, duet_path, ipi_xlsx_path, vis_params) {
  plots <- list()
  
  # 1. IPI Stratification KM Plot
  ipi_data <- read.csv(ipi_path)
  surv_object_ipi <- Surv(time = ipi_data$live.time, event = ipi_data$y)
  
  # Fit IPI KM model
  km_fit_ipi <- survfit(surv_object_ipi ~ IPI, data = ipi_data)
  
  # Log-rank test for IPI
  chi_square_ipi <- survdiff(surv_object_ipi ~ IPI, data = ipi_data)$chisq
  cat("IPI log-rank chi-square:", chi_square_ipi, "\n")
  
  # Generate IPI KM plot
  plots$ipi <- ggsurvplot(
    km_fit_ipi,
    data = ipi_data,
    break.time.by = 12,
    xlim = c(0, 90),
    palette = "Set2",
    risk.table = TRUE,
    pval = TRUE,
    conf.int = TRUE,
    title = "Kaplan - Meier Survival Curves by IPI",
    legend.labs = c("IPI=0or1", "IPI=2", "IPI=3", "IPI=4or5"),
    xlab = "Time",
    ylab = "Survival Probability",
    ggtheme = theme_minimal() + theme(plot.title = element_text(hjust = 0.5)),
    legend.justification = c("right", "top")
  )
  
  print(plots$ipi)
  print(summary(km_fit_ipi))
  
  # 2. Dual Expression (MYC) KM Plot
  duet_data <- read.csv(duet_path)
  ipi_xlsx <- read_xlsx(ipi_xlsx_path)
  
  # Merge duet data with IPI follow-up time
  duet_merged <- merge(duet_data, ipi_xlsx[, c("编号", "随访时间（m）")], by = "编号", all.x = TRUE)
  
  # Assuming MYC data is in 'myc' object (adjust column names as needed)
  # Note: Update column names to match your actual data structure
  myc <- duet_merged # Replace with actual MYC data if separate
  surv_object_myc <- Surv(time = myc$`随访时间（m）`, event = myc$PFS)
  
  # Fit MYC KM model
  km_fit_myc <- survfit(surv_object_myc ~ c.Myc, data = myc)
  
  # Log-rank test for MYC
  chi_square_myc <- survdiff(surv_object_myc ~ c.Myc, data = myc)$chisq
  cat("MYC dual expression log-rank chi-square:", chi_square_myc, "\n")
  
  # Generate MYC KM plot
  plots$myc_duet <- ggsurvplot(
    km_fit_myc,
    data = myc,
    break.time.by = 12,
    xlim = c(0, 60),
    palette = "#D55",
    risk.table = TRUE,
    pval = TRUE,
    conf.int = TRUE,
    title = "PFS Kaplan - Meier Survival Curves by Dual Expression",
    legend.labs = c(""),
    xlab = "Time",
    ylab = "Survival Probability",
    ggtheme = theme_minimal()
  )
  
  print(plots$myc_duet)
  print(summary(km_fit_myc))
  
  return(plots)
}

# ------------------------------------------------------------------------------
# Main Execution Function: Run All Analyses
# ------------------------------------------------------------------------------
#' Main Function to Execute All Survival Analysis and Visualization
#' 
#' Runs all modular functions in sequence and returns all plot objects
#' @param file_paths List of centralized file paths
#' @param vis_params List of visualization parameters
#' @return List of all generated plot objects
run_survival_analysis <- function(file_paths, vis_params) {
  # Store all plot results
  all_plots <- list()
  
  # Step 1: Plot OS vs PFS KM curves
  cat("=== Plotting OS vs PFS KM Curves ===\n")
  all_plots$os_pfs_km <- plot_os_pfs_km(
    os_path = file_paths$os_data,
    pfs_path = file_paths$pfs_data,
    vis_params = vis_params
  )
  
  # Step 2: Plot PFS predicted vs observed (test data)
  cat("\n=== Plotting PFS Test Predicted vs Observed KM Curves ===\n")
  all_plots$pfs_test_km <- plot_pfs_pred_obs_km(
    pfs_km_path = file_paths$pfs_km,
    vis_params = vis_params,
    is_validation = FALSE
  )
  
  # Step 3: Plot PFS predicted vs observed (validation data)
  cat("\n=== Plotting PFS Validation Predicted vs Observed KM Curves ===\n")
  all_plots$pfs_val_km <- plot_pfs_pred_obs_km(
    pfs_km_path = file_paths$pfs_km_validation,
    vis_params = vis_params,
    is_validation = TRUE
  )
  
  # Step 4: Plot similarity metrics (confusion matrix, scatter, Bland-Altman)
  cat("\n=== Plotting Similarity Metrics ===\n")
  # Load confusion matrix data (adjust path as needed)
  confusion_data <- read.csv(file_paths$pfs_km_validation) # Replace with actual confusion data
  all_plots$similarity <- plot_similarity_metrics(
    similarity_path = file_paths$os_similarity,
    bland_altman_path = file_paths$os_km_bland,
    confusion_data = confusion_data
  )
  
  # Step 5: Plot risk stratification (High/Low Risk) KM curves
  cat("\n=== Plotting Risk Stratification KM Curves ===\n")
  all_plots$risk_strat_km <- plot_risk_strat_km(
    dca_validation_path = file_paths$os_dca_validation,
    dca_os_path = file_paths$dca_os,
    vis_params = vis_params
  )
  
  # Step 6: Plot IPI and Dual Expression (MYC) KM curves
  cat("\n=== Plotting IPI and Dual Expression KM Curves ===\n")
  all_plots$ipi_duet_km <- plot_ipi_duet_km(
    ipi_path = file_paths$ipi_data,
    duet_path = file_paths$duet_data,
    ipi_xlsx_path = file_paths$ipi_xlsx,
    vis_params = vis_params
  )
  
  # Return all plots for further customization
  return(all_plots)
}

# ------------------------------------------------------------------------------
# Execute the Full Analysis Pipeline
# ------------------------------------------------------------------------------
# Run all survival analysis and visualization
all_results <- run_survival_analysis(
  file_paths = FILE_PATHS,
  vis_params = VIS_PARAMS
)

# Access individual plots with all_results$plot_name (e.g., all_results$os_pfs_km)
cat("\n=== All survival analysis and visualization completed ===")
