import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

def compute_error(y_true, y_pred):
    accuracy = np.mean(y_true == (y_pred > 0.5))
    variance = accuracy * (1 - accuracy)
    return variance

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    # 计算TP, TN, FP, FN
    tp = np.sum(y_label == 1)  # 正例数量
    tn = np.sum(y_label == 0)  # 负例数量
    total = tp + tn
    for thresh in thresh_group:
        # Treat All strategy predicts all as positive
        fp = total - tp  # 所有样本中非正样本数即假阳性数（因为预测全部为正样本）
        fn = 0  # 因为没有负样本被预测为负
        net_benefit = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'deepskyblue', label = 'train model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    #ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.1)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.2)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'lower left')

    return ax


def plot_DCA_test(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'orange', label = 'Internal Validation')
    ax.plot(thresh_group, net_benefit_all, color = 'slategrey',label = 'test all')
    # ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'T+LN Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    #ax.fill_between(thresh_group, y1, y2, color = 'navy', alpha = 0.1)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.2)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'lower left')

    return ax


def plot_DCA_external(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'green', label = 'validation')
    ax.plot(thresh_group, net_benefit_all, color = 'grey',label = 'validation treat all')
    # ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'T+LN Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    #ax.fill_between(thresh_group, y1, y2, color = 'navy', alpha = 0.1)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.2)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'lower left')

    return ax



def plot_calibration_curve(ax, fraction_of_positives, mean_predicted_value,error):
    ax.errorbar(mean_predicted_value, fraction_of_positives, yerr=error,color='red', label='Model',capsize=5, capthick=2,linewidth=1.5)
    ax.scatter(mean_predicted_value, fraction_of_positives, facecolors='none',  edgecolor='red', s=10, zorder=5)
    ax.plot((0, 1), (0, 1), color='black', linestyle=':')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)  # adjustify the y axis limitation
    ax.set_xlabel(
        xlabel='Predicted Value',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.set_ylabel(
        ylabel='Fraction of Positives',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper left')

    return ax


def plot_calibration_curve_test(ax, fraction_of_positives, mean_predicted_value,error):
    ax.errorbar(mean_predicted_value, fraction_of_positives,yerr = error, color='deepskyblue', label='Model_test',capsize=5, capthick=2,linewidth=1.5)
    ax.scatter(mean_predicted_value, fraction_of_positives, facecolors='none',  edgecolor='deepskyblue', s=10, zorder=5)
    ax.plot((0, 1), (0, 1), color='black', linestyle=':')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)  # adjustify the y axis limitation
    ax.set_xlabel(
        xlabel='Predicted Value',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.set_ylabel(
        ylabel='Fraction of Positives',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper left')

    return ax



def plot_calibration_curve_external(ax, fraction_of_positives, mean_predicted_value,error):
    ax.errorbar(mean_predicted_value, fraction_of_positives,yerr = error, color='green', label='Model_validation',capsize=5, capthick=2,linewidth=1.5)
    ax.scatter(mean_predicted_value, fraction_of_positives, facecolors='none',  edgecolor='green', s=10, zorder=5)
    ax.plot((0, 1), (0, 1), color='black', linestyle=':')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)  # adjustify the y axis limitation
    ax.set_xlabel(
        xlabel='Predicted Value',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.set_ylabel(
        ylabel='Fraction of Positives',
        fontdict={'family': 'Times New Roman', 'fontsize': 20}
    )
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper left')

    return ax






def combine_datasets_and_calculate_net_benefit_model(thresh_group, y_pred_score_dataset1, y_label_dataset1,
                                                     y_pred_score_dataset2, y_label_dataset2):
    y_pred_score_combined = np.concatenate((y_pred_score_dataset1, y_pred_score_dataset2))
    y_label_combined = np.concatenate((y_label_dataset1, y_label_dataset2))

    net_benefit_model_combined = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score_combined > thresh
        tn, fp, fn, tp = confusion_matrix(y_label_combined, y_pred_label).ravel()
        n = len(y_label_combined)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model_combined = np.append(net_benefit_model_combined, net_benefit)

    return net_benefit_model_combined, y_label_combined




if __name__ == '__main__':
    df = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\DCA\DCA OS.csv")
    df_test = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\DCA\DCA OS test.csv")
    # Convert DataFrame to NumPy array with column names
    npdf = df.to_numpy()
    npdf_test = df_test.to_numpy()
    y_pred_score = npdf[:,1]
    y_label = npdf[:,0]
    y_pred_score_test = npdf_test[:,1]
    y_label_test = npdf_test[:,0]
    error_train = np.std(y_pred_score)
    error_test = np.std(y_pred_score_test)

    y_pred_score_test = y_pred_score_test[~np.isnan(y_pred_score_test)]
    y_label_test = y_label_test[~np.isnan(y_label_test)]
    print(error_test)
    df_external = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\DCA\DCA validation.csv")
    npdf_external = df_external.to_numpy()
    y_pred_score_external = npdf_external[:, 1]  # 根据您的数据结构调整索引
    y_label_external = npdf_external[:, 0]        # 根据您的数据结构调整索引
    error_validation = np.std(y_pred_score_external)

    thresh_group = np.arange(0,1,0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    net_benefit_model_test = calculate_net_benefit_model(thresh_group, y_pred_score_test, y_label_test)
    net_benefit_all_test = calculate_net_benefit_all(thresh_group, y_label_test)
    net_benefit_model_external = calculate_net_benefit_model(thresh_group, y_pred_score_external, y_label_external)
    net_benefit_all_external = calculate_net_benefit_all(thresh_group, y_label_external)

    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
    ax = plot_DCA_test(ax, thresh_group, net_benefit_model_test, net_benefit_all_test)
    ax = plot_DCA_external(ax, thresh_group, net_benefit_model_external, net_benefit_all_external)  # 添加外部验证集的 DCA 曲线

    plt.show()

    fraction_of_positives, mean_predicted_value = calibration_curve(y_label, y_pred_score, pos_label=1, n_bins=4, strategy="quantile")
    fraction_of_positives_test, mean_predicted_value_test = calibration_curve(y_label_test, y_pred_score_test, pos_label=1, n_bins=4, strategy="quantile")
    fraction_of_positives_external, mean_predicted_value_external = calibration_curve(y_label_external, y_pred_score_external, pos_label=1, n_bins=4, strategy="quantile")

    fig_1, ax_1 = plt.subplots()
    ax_1 = plot_calibration_curve(ax_1, fraction_of_positives, mean_predicted_value,error_train)
    ax_1 = plot_calibration_curve_test(ax_1, fraction_of_positives_test, mean_predicted_value_test,error_test)
    ax_1 = plot_calibration_curve_external(ax_1, fraction_of_positives_external, mean_predicted_value_external,error_validation)  # 添加外部验证集的校准曲线

    plt.show()
