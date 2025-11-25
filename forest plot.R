
data = read.csv("D:/Acsq/BIM硕士第三学期/AD/补充/forest plot.csv")
data$CI_range = paste(data$exp.coef..lower.95.,data$exp.coef..upper95.,sep = "-")
data[1,14] = "CI_range"
data[,c(3,7,8)] = lapply(data[,c(3,7,8)],as.numeric)
data[, c(12, 10, 14)] <- lapply(data[, c(12, 10, 14)], as.character)
fig <- forestplot(
  data[, c(1, 12, 10, 14)],   # 选择显示的数据列
  mean = data[, 3],          # HR值
  lower = data[, 7],         # 95%置信区间的下限
  upper = data[, 8],         # 95%置信区间的上限
  zero = 1,                  # 参考线设置
  boxsize = 0.2,             # 小方块的大小
  graph.pos = 2,             # 森林图在图形中的位置
  hrzl_lines = list(
    "1" = gpar(lty = 1, lwd = 2),  # 均值线
    "2" = gpar(lty = 2),            # 下限和上限之间的虚线
    "36" = gpar(lwd = 2, lty = 1)   # 下限和上限线
  ),
  graphwidth = unit(.25, "npc"),
  txt_gp = fpTxtGp(
    label = gpar(cex = 0.7, fontface = "bold"),    # 标签的大小
    ticks = gpar(cex = 1),      # 刻度标记大小
    xlab = gpar(cex = 0.9),     # x轴标签的大小
    title = gpar(cex = 1.5, col = "darkred")  # 修改标题的大小和颜色
  ),
  lwd.zero = 1,                # 均值线的宽度
  lwd.ci = 1.5,                # 置信区间线的宽度
  lwd.xaxis = 2,               # x轴线的宽度
  lty.ci = 1.5,                # 置信区间线的样式
  ci.vertices = TRUE,          # 显示置信区间的顶点
  ci.vertices.height = 0.2,    # 置信区间顶点的高度
  clip = c(0.1, 8),            # x轴的范围
  ineheight = unit(8, 'mm'),   # 森林图内线的高度
  line.margin = unit(8, 'mm'),  # 森林图线的边距
  colgap = unit(6, 'mm'),       # 森林图中方框的间距
  fn.ci_norm = "fpDrawDiamondCI", # 置信区间的形状
  title = "Hazard Ratio",        # 自定义森林图标题
  col = fpColors(
    box = "blue4",              # 方框颜色
    lines = "blue4",            # 线条颜色
    zero = "black"              # 均值为0时的颜色
  ),
)

# 绘制森林图
print(fig)



library(ggplot2)
library(dplyr)
library(caret)

# 读取数据
df <- read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/DCA PFS.csv")
df_external <- read.csv("D:/Acsq/BIM硕士第三学期/AD/DCA/DCA PFS validation.csv")

# 定义绘制校准曲线的函数
plot_calibration_curve <- function(data, label_col, prob_col) {
  data <- data %>%
    # 将预测概率分为10个区间
    mutate(predicted_category = cut(!!sym(prob_col), 
                                    breaks = seq(0, 1, by = 0.1), 
                                    include.lowest = TRUE)) %>%
    group_by(predicted_category) %>%
    summarize(fraction_of_positives = mean(!!sym(label_col)),
              mean_predicted_value = mean(!!sym(prob_col)),
              count = n())
  
  # 绘制校准曲线
  ggplot(data, aes(x = mean_predicted_value, 
                   y = fraction_of_positives)) +
    geom_line(color = 'crimson', size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = 'dashed',
                color = 'black') +  # 对角线
    labs(x = 'Predicted Probability', 
         y = 'Fraction of Positives') +
    theme_minimal(base_size = 15) +
    theme(axis.text = element_text(family = 'Times New Roman'),
          axis.title = element_text(family = 'Times New Roman'),
          plot.title = element_text(family = 'Times New Roman'))
}

# 创建校准曲线图
par(mfrow = c(1, 2))  # 设置为2列1行布局

# 绘制训练数据的校准曲线
plot_calibration_curve(df, 'true_label', 'predicted_prob')
title(main = 'Calibration Curve - Training Data')

# 绘制验证数据的校准曲线
plot_calibration_curve(df_external, 'true_label', 'predicted_prob')
title(main = 'Calibration Curve - Validation Data')