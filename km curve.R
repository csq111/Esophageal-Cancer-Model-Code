library(survival)
library(survminer)
library(ggplot2) 
before_data_OS = read.csv("D:/Acsq/BIM硕士第三学期/AD/final data/EC_before_Treatment 7.12 OS.csv")
before_data_PFS = read.csv("D:/Acsq/BIM硕士第三学期/AD/final data/EC_before_Treatment-8.2 PFS.csv")
# 合并数据
combined_data <- rbind(
  data.frame(time = before_data_OS$OS_m, status = before_data_OS$OS, group = "OS"),
  data.frame(time = before_data_PFS$PFS_m, status = before_data_PFS$PFS, group = "PFS")
)


# 将时间转换为 10 个月的节点
max_time <- max(combined_data$time)
breaks_seq <- seq(0, max_time + 10, by = 10)

# 计算分组的数量
num_labels <- length(breaks_seq) - 1  # breaks 比 labels 多 1

# 创建标签
labels_seq <- seq(0, by = 10, length.out = num_labels)

# 将时间转换为 10 个月的区间
combined_data$time_group <- cut(combined_data$time, 
                                breaks = breaks_seq, 
                                right = FALSE,  
                                labels = labels_seq)

# 创建生存对象
surv_object <- Surv(time = combined_data$time, event = combined_data$status)

# 进行 Kaplan-Meier 生存分析
km_fit <- survfit(surv_object ~ group, data = combined_data)

# 创建核对风险人数的函数，以确保风险表中的内容准确
get_risk_table <- function(model, data, breaks) {
  risk_counts <- data.frame(time = breaks[-length(breaks)], n.risk = NA)
  
  for (i in 1:(length(breaks) - 1)) {
    risk_counts$n.risk[i] <- sum(data$time >= risk_counts$time[i])
  }
  
  return(risk_counts)
}

# 取得风险表的内容
risk_table_data <- get_risk_table(km_fit, combined_data, breaks_seq)

# 绘制 KM 曲线及风险表
plot <- ggsurvplot(
  km_fit,
  data = combined_data,
  risk.table = TRUE,               # 添加风险表
  risk.table.title = "Number at Risk",
  xlab = "Time (Months)", 
  ylab = "Survival Probability", 
  title = "Kaplan-Meier Curves for OS and PFS",
  palette = c("#E7B800", "#2E9FDF"),   # 自定义颜色
  legend.title = "Group",          # 图例标题
  legend.labs = c("Overall Survival (OS)", "Progression-Free Survival (PFS)"),
  conf.int = TRUE,                  # 显示置信区间
  censor.shape = 3,                 # 审查点形状
  censor.size = 2                   # 审查点大小
)

# 修改 KM 图形的主题和 x 轴刻度
plot$plot <- plot$plot +
  theme_minimal(base_size = 16) + 
  theme(plot.title    = element_text(hjust = 0.5, size = 20, face = "bold"),
        legend.position = "top",
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid.major = element_line(linewidth = 0.5),
        panel.grid.minor = element_blank()
  ) +
  scale_x_continuous(breaks = seq(0, max_time, by = 10))  # 设置 x 轴为每 10 个月

# 修改风险表内容
plot$table <- ggplot(risk_table_data, aes(x = time, y = n.risk)) +
  geom_col() +
  scale_x_continuous(breaks = seq(0, max_time, by = 10)) +  # 更新风险表的 x 轴刻度为每 10 个月
  labs(x = "Time (Months)", y = "Number at Risk") +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

# 打印图形
print(plot)



# 风险表
combined_data <- rbind(
  data.frame(time = before_data_OS$OS_m, status = before_data_OS$OS, group = "OS"),
  data.frame(time = before_data_PFS$PFS_m, status = before_data_PFS$PFS, group = "PFS")
)

# 创建生存对象
surv_object <- Surv(time = combined_data$time, event = combined_data$status)

# 进行 Kaplan-Meier 生存分析
km_fit <- survfit(surv_object ~ group, data = combined_data)

lancet_colors <- c("#CC79A7", "#0072B2")  # 柳叶刀颜色方案

# 修改绘图代码
plot <- ggsurvplot(
  km_fit,
  data = combined_data,
  break.time.by = 10,
  risk.table = TRUE,               # 添加风险表
  risk.table.title = "",
  xlab = "Time (Months)", 
  ylab = "Survival Probability",
  title = "Kaplan-Meier Curves for OS and PFS",
  palette = lancet_colors,         # 使用柳叶刀风格的颜色方案
  legend.title = "Group",
  legend.labs = c("OS", "PFS"),   # 更新图例标签
  conf.int = TRUE,
  censor.shape = 3,
  censor.size = 2,
  xlim = c(0, 70),                 # 设置x轴的范围最大值为70
  ggtheme = theme_minimal(base_size = 16) +
    theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
          legend.position = "right",
          legend.title = element_blank(),
          legend.text = element_text(size = 12,color = "black"),
          axis.title.x = element_text(size = 14,color = "black"),
          axis.title.y = element_text(size = 14),
          axis.text.x = element_text(size = 12,color = "black"),
          axis.text.y = element_text(size = 12),
          panel.grid.major = element_line(size = 0.5),
          panel.grid.minor = element_blank())
)

# 添加x轴的刻度间隔设置
plot$plot <- plot$plot + scale_x_continuous(breaks = seq(0, 70,  10))

# 打印图形
print(plot)




library(survival)
library(survminer)
library(ggplot2) 
data = read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/PFS KM.csv")

combined_data <- rbind(
  data.frame(time =data$live.time.test, status = data$y.test, group = "EV True"),
  data.frame(time = data$prediction.time.test, status = data$y.test.predictions, group = "EV Prediction")
)
# 创建生存对象
surv_object <- Surv(time = combined_data$time, event = combined_data$status)

# 进行 Kaplan-Meier 生存分析
km_fit <- survfit(surv_object ~ group, data = combined_data)

lancet_colors <- c("#CC79A7", "#0072B2")  # 柳叶刀颜色方案

# 修改绘图代码
plot <- ggsurvplot(
  km_fit,
  data = combined_data,
  break.time.by = 10,
  risk.table = TRUE,               # 添加风险表
  risk.table.title = "",
  xlab = "Time (Months)", 
  ylab = "Progression Free Survival",
  title = "Kaplan-Meier Curves for True and Prediction",
  palette = lancet_colors,         # 使用柳叶刀风格的颜色方案
  legend.title = "",
  legend.labs = c("Observed", "Predicted"),   # 更新图例标签
  conf.int = TRUE,
  censor.shape = 3,
  censor.size = 2,
  xlim = c(0, 70),                 # 设置x轴的范围最大值为70
  ggtheme = theme_minimal(base_size = 16) +
    theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
          legend.position = "right",
          legend.title = element_blank(),
          legend.text = element_text(size = 14,color = "black"),
          axis.title.x = element_text(size = 14,color = "black"),
          axis.title.y = element_text(size = 14),
          axis.text.x = element_text(size = 14,color = "black"),
          axis.text.y = element_text(size = 14),
          panel.grid.major = element_line(size = 0.5),
          panel.grid.minor = element_blank())
)

# 添加x轴的刻度间隔设置
plot$plot <- plot$plot + scale_x_continuous(breaks = seq(0, 70,  10))

# 打印图形
print(plot)
# 提取 Kaplan-Meier 生存分析结果
km_summary <- summary(km_fit)

print(km_summary)


Progression-Free-Survival
##############################################################
library(ggplot2)
library(survival)
library(survminer)

# 读取数据
data = read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/PFS km.csv")

# 合并数据
combined_data <- rbind(
  data.frame(time = data$live.time.validation, status = data$y_validation, group = "Train True"),
  data.frame(time = data$prediction.time.validation, status = data$y_validation_prediction, group = "Train Prediction")
)

# 创建生存对象
surv_object <- Surv(time = combined_data$time, event = combined_data$status)


# 进行 Kaplan-Meier 生存分析
km_fit <- survfit(surv_object ~ group, data = combined_data)

# 柳叶刀颜色方案
lancet_colors <- c("#CC79A7", "#0072B2")

# 修改绘图代码
# 柳叶刀颜色方案
lancet_colors <- c("#D55E00", "#56B4E9")

# 创建生存曲线图形
plot <- ggsurvplot(
  km_fit,
  data = combined_data,
  break.time.by = 10,
  risk.table = TRUE,
  pval = TRUE,  
  pval.method = TRUE,  
  pval.method.text = "logrank p", 
  risk.table.title = "",
  risk.table.fontsize = 13,
  risk.table.text = element_text(family = "Times New Roman"),
  xlab = "Time(month)", 
  ylab = "Progression Free Survival",
  title = "",
  palette = lancet_colors,         # 使用柳叶刀风格的颜色方案
  legend.title = "",
  legend.labs = c("Observed", "Predicted"),   
  conf.int = TRUE,
  censor.shape = 3,
  censor.size = 2,
  xlim = c(0, 70),                 # 设置 x 轴的范围最大值为 70
  ggtheme = theme_minimal(base_size = 20) +
    theme(
      text = element_text(family = "Times New Roman"),
      plot.title = element_text(hjust = 1, size = 25, face = "bold"),
      plot.subtitle = element_text(size = 40, face = "bold"),  
      legend.position = c(1, 1),  # 图例在右上角
      legend.justification = c("right", "top"),  # 图例的对齐方式
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

# 绘制图形
print(plot)


#similarity plot
confusion_matrix <- table(data$y.validation, data$validation.predictions)

confusion_df <- as.data.frame(confusion_matrix)
colnames(confusion_df) <- c("Actual", "Predicted", "Frequency")

# 绘制混淆矩阵图
ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

# 2. 绘制预测时间与实际时间的比较
library(extrafont)
# 计算相关系数
plotdata = read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/OS similarity plot - EV.csv")
test=plotdata$live.time.validation
prediction=plotdata$prediction.time.validation
label=plotdata$y.validation
correlation <- cor(test, prediction)

# 创建散点图并添加相关系数
p <- ggplot(plotdata, aes(x = test, y = prediction)) +
  geom_point(aes(color = as.factor(label)), size = 1) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +  # 添加线性回归线
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Prediction Time vs Live Time",
       x = "Prediction Time",
       y = "Live Time") +
  scale_color_manual(values = c("red", "blue"), name = "") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman", size = 15),  
        plot.title = element_text(hjust = 0.5, size = 15),
        legend.spacing.y = unit(0.5, "cm"), 
        legend.position = c(0.8, 0.1), 
        plot.background = element_rect(fill = "white")) +
  xlim(0, 80) +
  ylim(0, 80) +
  annotate("text", x = 10, y = 70, 
           label = paste("Correlation: ", round(correlation, 2)), 
           size = 5, color = "black", family = "Times New Roman")  # 设置注释字体大小
plot=p
print(plot)

# 保存图像为 1.97 x 1.97 英寸
#ggsave("D:/Acsq/BIM_last_term/其他/new figs2/PFS EV similarity plot.png", plot = p, width = 1.97, height = 2.3, units = "in", dpi = 1200)

#bland-altman plot
data=read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/OS KM.csv")
data$mean_train = rowMeans(data[,c('prediction.time.train','live.time.train')])
data$difference_train = data$prediction.time.train-data$live.time.train
mean_train=mean(data$difference_train)
sd_train=sd(data$difference_train)
ggplot(data, aes(x = mean_train, y = difference_train)) +
  geom_point(size=1,shape=1,color='orangered') +
  geom_hline(yintercept = mean_train, color = "blue", linetype = "dashed") +  # 均值差异线
  geom_hline(yintercept = mean_train + 1.96 * sd_train, color = "red", linetype = "dashed") +  # 上标准差线
  geom_hline(yintercept = mean_train - 1.96 * sd_train, color = "red", linetype = "dashed") +  # 下标准差线
  labs(title = "Bland-Altman Plot",
       x = "Mean of Measurements",
       y = "Difference between Measurements") +
  theme_minimal()

#error plot
error <- data$live.time.train - data$prediction.time.train
ggplot(data.frame(error), aes(x = error)) +
  geom_histogram(binwidth = 0.5,color='blue') +
  labs(title = "Error Distribution",
       x = "Error (True - Predicted)",
       y = "Frequency") +
  theme_minimal()


#######################################################

library(survival)
library(survminer)
library(ggplot2) 
data = read.csv("D:/Acsq/BIM硕士第三学期/AD/final data/EC_before_Treatment 7.12 OS 1.csv")

combined_data <- rbind(
  data.frame(time =data$OS_m, status = data$OS),
  #data.frame(time = data$prediction.time.validation, status = data$validation.predictions, group = "EV Prediction")
)
# 创建生存对象
surv_object <- Surv(time = combined_data$time, event = combined_data$status)

# 进行 Kaplan-Meier 生存分析
km_fit <- survfit(surv_object ~ group, data = combined_data)

lancet_colors <- c("#CC79A7", "#0072B2")  # 柳叶刀颜色方案

# 修改绘图代码
plot <- ggsurvplot(
  km_fit,
  data = combined_data,
  break.time.by = 10,
  risk.table = TRUE,               # 添加风险表
  risk.table.title = "",
  xlab = "Time (Months)", 
  ylab = "Overall Survival",
  title = "Kaplan-Meier Curves for True and Prediction",
  palette = lancet_colors,         # 使用柳叶刀风格的颜色方案
  legend.title = "",
  legend.labs = c("EV True", "EV Prediction"),   # 更新图例标签
  conf.int = TRUE,
  censor.shape = 3,
  censor.size = 2,
  xlim = c(0, 70),                 # 设置x轴的范围最大值为70
  ggtheme = theme_minimal(base_size = 16) +
    theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
          legend.position = "right",
          legend.title = element_blank(),
          legend.text = element_text(size = 14,color = "black"),
          axis.title.x = element_text(size = 14,color = "black"),
          axis.title.y = element_text(size = 14),
          axis.text.x = element_text(size = 14,color = "black"),
          axis.text.y = element_text(size = 14),
          panel.grid.major = element_line(size = 0.5),
          panel.grid.minor = element_blank())
)

# 添加x轴的刻度间隔设置
plot$plot <- plot$plot + scale_x_continuous(breaks = seq(0, 70,  10))

# 打印图形
print(plot)
# 提取 Kaplan-Meier 生存分析结果
km_summary <- summary(km_fit)

print(km_summary)

library(ggplot2)
library(survival)
library(survminer)

# 读取数据
data <- read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/OS DCA validation.csv")
train_data = read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/DCA OS.csv")
data$risk_group <- ifelse(data$validation.predictions >= median(train_data$y.train.predictions), "High Risk", "Low Risk")
print(median(train_data$y.train.predictions))
print(median(data$y.test.predictions))
surv_object <- Surv(time = data$live.time, event = data$y.validation)
km_high_risk <- survfit(surv_object ~ data$risk_group == "High Risk")
km_low_risk <- survfit(surv_object ~ data$risk_group == "Low Risk")
km_fit <- survfit(surv_object ~ risk_group, data = data)
chi_square_result = survdiff(surv_object ~ risk_group, data = data)
chi_square = chi_square_result$chisq
print(chi_square)
ggsurvplot(
  km_fit,
  data = data,
  risk.table = TRUE,        # 显示风险表
  pval = TRUE,              # 显示 p 值
  conf.int = TRUE,          # 显示置信区间
  ggtheme = theme_minimal(), # 使用简约主题
  title = "Kaplan-Meier Survival Curve",
  xlab = "Time",
  ylab = "Survival Probability"
)
# 整合数据
#km_data <- rbind(data.frame(time = data$OS_m_1, status = data$OS_1, group = "1"),
                 #data.frame(time = data$OS_m_2, status = data$OS_2, group = "2"))

# 创建生存对象
#surv_object <- Surv(time = km_data$time, event = km_data$status)

# 拟合Kaplan-Meier曲线
#km_fit <- survfit(surv_object ~ group, data = data)

# 柳叶刀颜色方案
lancet_colors <- c("#D55E00", "#56B4E9")

# 创建生存曲线图形
plot <- ggsurvplot(
  km_fit,
  data = data,
  break.time.by = 10,
  risk.table = TRUE,
  pval = TRUE,  
  pval.method = TRUE,  
  pval.method.text = "logrank p", 
  risk.table.title = "",
  risk.table.fontsize = 13,
  risk.table.text = element_text(family = "Times New Roman"),
  xlab = "Time(month)", 
  ylab = "Progression Free Survival",
  title = "",
  palette = lancet_colors,         # 使用柳叶刀风格的颜色方案
  legend.title = "",
  legend.labs = c("High Risk", "Low Risk"),   # 更新图例标签
  conf.int = TRUE,
  censor.shape = 3,
  censor.size = 2,
  xlim = c(0, 70),                 # 设置 x 轴的范围最大值为 70
  ggtheme = theme_minimal(base_size = 20) +
    theme(
      text = element_text(family = "Times New Roman"),
      plot.title = element_text(hjust = 1, size = 25, face = "bold"),
      plot.subtitle = element_text(size = 40, face = "bold"),  
      legend.position = c(1, 1),  # 图例在右上角
      legend.justification = c("right", "top"),  # 图例的对齐方式
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

# 绘制图形
print(plot)
km_summary <- summary(km_fit)
print(km_summary)

###############################################
data = read.csv("C:/Users/Lenovo/Desktop/zyy/km/统计结果/IPI.csv")
#IPI分组
num_samples <- nrow(data)
group_indices <- rep(1:5, length.out = num_samples)
surv_object <- Surv(time = data$live.time, event = data$y)
data$risk_group <- ifelse(data$validation.predictions >= median(train_data$y.train.predictions), "High Risk", "Low Risk")
surv_object <- Surv(time = data$live.time, event = data$y.validation)
km_1_risk <- survfit(surv_object ~ data$IPI <= 1)
km_2_risk <- survfit(surv_object ~ data$IPI == 2)
km_3_risk <- survfit(surv_object ~ data$IPI == 3)
km_4_risk <- survfit(surv_object ~ data$IPI == 4)
km_5_risk <- survfit(surv_object ~ data$IPI == 5)
#high low risk分组
km_high_risk <- survfit(surv_object ~ data$risk_group == "High Risk")
km_low_risk <- survfit(surv_object ~ data$risk_group == "Low Risk")
#不分组
km_fit <- survfit(surv_object ~ IPI, data = data)
#卡方检验
chi_square_result = survdiff(surv_object ~ IPI, data = data)
chi_square = chi_square_result$chisq
print(chi_square)
#画图
ggsurvplot(
  km_fit,
  data = data,
  break.time.by = 12,
  xlim = c(0, 90),
  palette = "#D55E00", 
  risk.table = TRUE,        # 显示风险表
  pval = TRUE,              # 显示 p 值
  conf.int = TRUE,          # 显示置信区间
  title = "Overall Survival Kaplan-Meier Survival Curve",
  xlab = "Time",
  ylab = "Probability",
  ggtheme = theme_minimal() + theme(plot.title = element_text(hjust = 0.5)),
  legend.justification = c("right","top")
)

#分组画图
ggsurvplot(
  km_fit,
  data = data,
  break.time.by = 12,  # 根据时间间隔分割x轴，可按需调整
  xlim = c(0, max(data$live.time)),  # 设置x轴范围
  palette = "Set2",  # 选择配色方案，可按需更换
  risk.table = TRUE,  # 显示风险表
  pval = TRUE,  # 显示组间差异的p值
  conf.int = TRUE,  # 显示置信区间
  ggtheme = theme_minimal(),  # 使用简约主题
  title = "Kaplan - Meier Survival Curves by IPI",
  legend.labs = c("IPI=0or1", "IPI=2","IPI=3","IPI=4or5"),
  xlab = "Time",
  ylab = "Survival Probability"
)
# 绘制图形
print(plot)
#打印概括
km_summary <- summary(km_fit)
print(km_summary)


###################################
#MYC and duet
data = read.csv("C:/Users/Lenovo/Desktop/zyy/program(1)/program/双表达/duet.csv")
ipi = read_xlsx("C:/Users/Lenovo/Desktop/zyy/program(1)/program/ipi/ipi.xlsx")
duet <- merge(data, ipi[, c("编号", "随访时间（m）")], by = "编号", all.x = TRUE)
#IPI分组
num_samples <- nrow(data)
group_indices <- rep(1:5, length.out = num_samples)
surv_object <- Surv(time = myc$`随访时间（m）`, event = myc$PFS)
data$risk_group <- ifelse(data$validation.predictions >= median(train_data$y.train.predictions), "High Risk", "Low Risk")
surv_object <- Surv(time = data$live.time, event = data$y.validation)
km_1_risk <- survfit(surv_object ~ data$IPI <= 1)
km_2_risk <- survfit(surv_object ~ data$IPI == 2)
km_3_risk <- survfit(surv_object ~ data$IPI == 3)
km_4_risk <- survfit(surv_object ~ data$IPI == 4)
km_5_risk <- survfit(surv_object ~ data$IPI == 5)
#high low risk分组
km_high_risk <- survfit(surv_object ~ data$risk_group == "High Risk")
km_low_risk <- survfit(surv_object ~ data$risk_group == "Low Risk")
#不分组
km_fit <- survfit(surv_object ~ c.Myc, data = myc)
#卡方检验
chi_square_result = survdiff(surv_object ~ c.Myc, data = myc)
chi_square = chi_square_result$chisq
print(chi_square)
#画图
ggsurvplot(
  km_fit,
  data = data,
  break.time.by = 12,
  xlim = c(0, 90),
  palette = "#D55E00", 
  risk.table = TRUE,        # 显示风险表
  pval = TRUE,              # 显示 p 值
  conf.int = TRUE,          # 显示置信区间
  title = "Overall Survival Kaplan-Meier Survival Curve",
  xlab = "Time",
  ylab = "Probability",
  ggtheme = theme_minimal() + theme(plot.title = element_text(hjust = 0.5)),
  legend.justification = c("right","top")
)

#分组画图
ggsurvplot(
  km_fit,
  data = data,
  break.time.by = 12,  # 根据时间间隔分割x轴，可按需调整
  xlim = c(0, 60),  # 设置x轴范围
  palette = "#D55",  # 选择配色方案，可按需更换
  risk.table = TRUE,  # 显示风险表
  pval = TRUE,  # 显示组间差异的p值
  conf.int = TRUE,  # 显示置信区间
  ggtheme = theme_minimal(),  # 使用简约主题
  title = "PFS Kaplan - Meier Survival Curves by Dual Expression",
  legend.labs = c(''),
  xlab = "Time",
  ylab = "Survival Probability"
)
# 绘制图形
print(plot)
#打印概括
km_summary <- summary(km_fit)
print(km_summary)