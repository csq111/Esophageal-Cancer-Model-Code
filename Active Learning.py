import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, mutual_info_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from catboost import Pool
from catboost import CatBoost
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, RFECV, RFE
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import shap
import math

data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/before OS 7.19.csv").dropna()
final_data_0S_before = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 OS after preprocessing.csv").dropna()

z_score = stats.zscore(data)
threshold = 4
outliers = (z_score > threshold).any(axis=1)
data_no_outliers = data[~outliers]
data_no_outliers = data_no_outliers.dropna()


threshold = 0.8
corr_matrix = data_no_outliers.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
X_final = data_no_outliers.drop(to_drop, axis=1)

print(X_final)
X = X_final.drop('OS',axis=1)
X = X.drop('OS_m',axis=1)

y = X_final['OS']

scaler =StandardScaler()
keywords = ['treatment', 'TL', 'original']
columns_to_standardize = [col for col in X.columns if any(keyword in col for keyword in keywords)]
X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])


X = X.astype(str)
y = y.astype(str)
X_train, X_test,y_train,y_test = train_test_split(X , y ,test_size=0.2, random_state=42)
print(X_test)
X_test_final = X_final.drop(X_train.index)
print(X_test_final)
print(X_train)

#X_test_final.to_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.2 OS.csv", index=False)
class LoglossObjective(object):
    def __init__(self, penalty=1.3,alpha = 0.5,mrf_weight = 0.5,reward_factor = 1.8):
        self.alpha = alpha
        self.penalty = penalty
        self.mrf_weight = mrf_weight
        self.reward_factor = reward_factor
    def stable_log(self, x, eps=1e-10):
        # 使用 log1p 来确保数值稳定性
        return np.log1p(x + eps) if x > 1 - eps else np.log(x + eps)

    def adjust_factors_based_on_performance(self, validation_performance):
        # 根据验证集上的性能动态调整奖励因子和惩罚项
        if validation_performance.decreased:
            self.reward_factor *= 1.1  # 性能下降，增加奖励因子
            self.penalty *= 1.1        # 增加惩罚项
        else:
            self.reward_factor *= 0.9  # 性能提升，减少奖励因子
            self.penalty *= 0.9
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = [np.exp(a) for a in approxes]  # 使用列表推导式提高效率
        gamma = 2
        beta = 0.5

        result = []
        for index in range(len(targets)):

            p = exponents[index] / (1 + exponents[index])
            # 计算 penalty，当 p 等于 0.5 时，不应用任何惩罚
            penalty = 1.0
            if (targets[index] == 1 and p < 0.2) or (targets[index] == 0 and p > 0.8):
                penalty *= self.penalty
            # 初始化 reward_factor 为 1，以便在 p 不等于 0.5 时进行调整
            reward_factor = 1.0
            # 根据 p 和目标值调整 reward_factor
            if targets[index] == 1:
                if p > 0.8:
                    reward_factor *= self.reward_factor
            elif targets[index] == 0:
                if p < 0.2:
                    reward_factor *= self.reward_factor

            der1_log = (1-p) * self.penalty  if targets[index] > 0.0 else -p * self.penalty
            der2_log = -p * (1-p)

            der1_mse = 2 * (p - targets[index])
            der2_mse = 2
            # mrf_penalty = self.mrf_weight * (approxes[index] - approxes[index - 1]) ** 2 if index > 0 else 0
            delta = 0.1 # Huber loss parameter
            # 计算 Huber 损失的一阶导数
            der1_huber = (targets[index] - p) if abs(targets[index] - p) <= delta else np.sign(targets[index] - p) * delta
            # 计算 Huber 损失的二阶导数
            der2_huber = 1 if abs(targets[index] - p) <= delta else 0
            # 添加Hinge Loss的计算
            margin = 1 - targets[index] * approxes[index]
            der1_hinge = -targets[index] if margin < 0 else 0
            der2_hinge = 0

            q=0.5
            quantile_loss = q * np.abs(targets[index] - p) if targets[index] - p >= 0 else (1 - q) * np.abs(targets[index] - p)
            der1_quantile = q * (p-p**2) if targets[index] - p >= 0 else (q-1) * (p-p**2)
            der2_quantile = 0

            if targets[index] > 0.0:
                focal_loss = -beta * (1 - p) ** gamma * np.log(p + 1e-40)
            else:
                focal_loss = -((1 - beta) * p) ** gamma * np.log(1 - p + 1e-40)
            if targets[index] > 0.0:
                der1_focal = beta * gamma * (1 - p) ** (gamma - 1) * np.log(p+1e-40) - beta * (1-p) ** gamma  * 1/(p+1e-40)
            else:
                der1_focal = -gamma * p **(gamma-1) * np.log(1-p+1e-40) + p**gamma*1/(1-p+1e-40)
            der2_focal = 0

            class_frequency = np.mean(targets)
            weight_factor = 1 / class_frequency if targets[index]  == 1 else class_frequency


            der1_combined = (der1_quantile+der1_log) / 2
            der2_combined = (der2_quantile+der2_log) / 2


            if weights is not None:
                der1_combined *= weights[index]
                der2_combined *= weights[index]



            result.append((der1_combined, der2_combined))

        return result

# catboost_model = CatBoostClassifier(cat_features=['Age','Location','T','N','TNM','PTV_Dose','GTV_Dose'])# 定义要调优的参数网格
# param_grid = {
#         'depth': [4, 6, 8],
#         'learning_rate': [0.01,0.02, 0.001,0.003, 0.1,0.2],
#         'iterations': [400, 600,800,1000, 1200]
# }
# # 使用 Grid Search 和交叉验证进行参数搜索
# grid_search = GridSearchCV(catboost_model, param_grid, cv=3)
# grid_search.fit(X_train, y_train)
# # 输出最佳参数组合
# best_params = grid_search.best_params_
# print("最佳参数组合：", best_params)
# # 使用最参数重新训练模型
# best_catboost_model = CatBoostClassifier(cat_features=['Age', 'Location', 'T', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose'], **best_params)
# best_catboost_model.fit(X_train, y_train)
# # 使用最佳参数重新训练模型
# best_catboost_model = CatBoostClassifier(cat_features=['Age', 'Location', 'T', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose'], **best_params)
# best_catboost_model.fit(X_train, y_train)
# # 获取训练集和测试集的预测概率
# train_probs = best_catboost_model.predict_proba(X_train)[:, 1]
# test_probs = best_catboost_model.predict_proba(X_test)[:, 1]
# # 计算 ROC 曲线的各项指标
# train_fpr, train_tpr, _ = roc_curve(y_train, train_probs, pos_label='1')  # 明确指定正类标签为 '1'
# test_fpr, test_tpr, _ = roc_curve(y_test, test_probs, pos_label='1')  # 明确指定正类标签为 '1'
# # 计算 ROC 曲线下面积（AUC）
# train_auc = roc_auc_score(y_train, train_probs)
# test_auc = roc_auc_score(y_test, test_probs)
# # 画出 ROC 曲线
# plt.figure(figsize=(8, 6))
# plt.plot(train_fpr, train_tpr, label=f'Train AUC = {train_auc:.2f}')
# plt.plot(test_fpr, test_tpr, label=f'Test AUC = {test_auc:.2f}')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()


def active_learning_sample_selection_with_blad(X_train, y_train, initial_labeled_samples=50, num_iterations=32, num_samples_to_label=10):
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 选择初始标记样本
    X_labeled = X_train[:initial_labeled_samples]
    y_labeled = y_train[:initial_labeled_samples]
    model1 = CatBoostClassifier(cat_features=list(range(13)), loss_function=LoglossObjective(), eval_metric='Logloss', depth=8, iterations=400, learning_rate=0.03)
    model2 = RandomForestClassifier(n_estimators=100)

    for i in range(num_iterations):
        # 未标记样本
        X_unlabeled = X_train[len(X_labeled):]
        y_unlabeled = y_train[len(y_labeled):]
        model1.fit(X_labeled, y_labeled)
        model2.fit(X_labeled, y_labeled)
        # 获取未标记样本的预测概率
        predictions1 = model1.predict_proba(X_unlabeled)
        predictions2 = model2.predict_proba(X_unlabeled)
        # 计算每个样本的不一致性
        disagreements = np.abs(predictions1 - predictions2).mean(axis=1)
        # 选择不一致性最高的样本
        selected_indices = np.argsort(disagreements)[-num_samples_to_label:]
        # 标记选定的样本并加入标记样本集
        X_labeled = np.concatenate((X_labeled, X_unlabeled[selected_indices]))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[selected_indices]))
        # 在每次迭代后重新训练模型
        model1.fit(X_labeled, y_labeled)
        model2.fit(X_labeled, y_labeled)

    # 计算被筛选掉的样本
    removed_indices = set(range(len(X_train))) - set(range(len(X_labeled)))

    # 获取被筛选掉的样本及其标签
    X_removed = X_train[list(removed_indices)]
    y_removed = y_train[list(removed_indices)]

    # 将最终结果转换为 Pandas DataFrame
    X_labeled_bald = pd.DataFrame(X_labeled)
    y_labeled_bald = pd.DataFrame(y_labeled)

    return X_labeled_bald, y_labeled_bald, X_removed, y_removed




X_labeled_bald, y_labeled_bald,X_removed,y_removed = active_learning_sample_selection_with_blad(X_train, y_train)

removed_sample = pd.concat([pd.DataFrame(X_removed), pd.DataFrame(y_removed)], axis=1)
# removed_sample.to_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment removed 42samples initial50.csv")
sample_labeld_bald = pd.concat([X_labeled_bald,y_labeled_bald],axis=1)
# sample_labeld_bald.to_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment remained 450samples initial50.csv")

# 提取标记的样本
labeled_samples = X_final.iloc[X_labeled_bald.index]

# 提取未标记的样本
unlabeled_samples = X_final.drop(labeled_samples.index)
#unlabeled_samples_22 = unlabeled_samples.drop(y_test.index)
# 将结果保存到CSV文件
labeled_samples.to_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.2 LC.csv", index=False)
unlabeled_samples.to_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment unlabeled 8.2 LC.csv", index=False)
