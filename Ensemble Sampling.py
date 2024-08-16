import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import preprocessing
from tabulate import tabulate
import seaborn as sns

class LoglossObjective(object):
    def __init__(self, penalty=2,alpha = 0.5,mrf_weight = 0.5,reward_factor = 0.9):
        self.alpha = alpha
        self.penalty = penalty
        self.mrf_weight = mrf_weight
        self.reward_factor = reward_factor
    def stable_log(self, x, eps=1e-10):
        # 使用 log1p 来确保数值稳定性
        return np.log1p(x + eps) if x > 1 - eps else np.log(x + eps)

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

            der1_log = (1-p) *self.penalty   if targets[index] > 0.0 else -p *self.penalty
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

            q=0.6
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


            der1_combined = (der1_log+der1_quantile) / 2
            der2_combined = (der2_log+der2_quantile) / 2


            if weights is not None:
                der1_combined *= weights[index]
                der2_combined *= weights[index]



            result.append((der1_combined, der2_combined))

        return result
def add_noise_to_labels(y_train, noise_level):
    """
    Add noise to the labels of the training set.

    Parameters:
        y_train (numpy array): True labels of the training set.
        noise_level (float): Percentage of labels to flip (e.g., 0.1 for 10% noise).

    Returns:
        noisy_labels (numpy array): Labels with added noise.
    """
    num_samples = len(y_train)
    num_noise_samples = int(noise_level * num_samples)

    # Randomly select indices to flip
    noise_indices = np.random.choice(num_samples, num_noise_samples, replace=False)

    # Flip the labels at selected indices
    noisy_labels = np.copy(y_train)
    noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]  # Flipping labels

    return noisy_labels

train_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.2 PFS.csv").dropna()
test_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.1 PFS.csv").dropna()
validation = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 PFS external validation.csv")
data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 OS after preprocessing.csv").dropna()
drop_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_unlabeled_samples 22 7.22.csv").dropna()



train_all = pd.concat([train_data,drop_data],axis=0)
train_all_X = train_all.drop(['OS','OS_m'],axis=1).astype(str)
train_all_y = train_all['OS']

X_train = train_data.drop(['PFS','PFS_m'],axis=1).astype(str)
y_train = train_data['PFS']
X_test = test_data.drop(['PFS','PFS_m'],axis=1).astype(str)
y_test = test_data['PFS']
X_validation = validation.drop(['PFS','PFS_m'],axis=1).astype(str)
y_validation = validation['PFS']
y_train_noisy = add_noise_to_labels(y_train, noise_level=0.05)
y_test_noisy = add_noise_to_labels(y_test,noise_level=0.01)
y_validation_noisy = add_noise_to_labels(y_validation,noise_level=0.01)
X_test_final = train_data.iloc[X_test.index]
print(X_test_final)
# test_data = pd.concat([X_test,y_test],axis=1)
# test_data.to_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment test data 7.24.csv", index=False)
scaler =StandardScaler()
keywords = ['treatment', 'TL', 'original']
columns_to_standardize = [col for col in X_train.columns if any(keyword in col for keyword in keywords)]
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.fit_transform(X_test[columns_to_standardize])
X_validation[columns_to_standardize] = scaler.fit_transform(X_validation[columns_to_standardize])

# 设置权重
weights = pd.Series([3] * 370 + [2] * 22)  # 假设370例为3的权重，22例为2的权重
weights = weights / weights.sum()  # 归一化权重

train_pool = Pool(X_train,y_train,cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])
test_pool = Pool(X_test,y_test,cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])


print(X_train.columns)
print(X_test.columns)
print(train_pool.get_feature_names())
print(test_pool.get_feature_names())
catboost_model = CatBoostClassifier(depth=8, iterations=400, learning_rate=0.1,early_stopping_rounds = 50,loss_function = LoglossObjective(),eval_metric = 'Logloss', cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'],random_seed=42)
catboost_model.fit(train_pool,eval_set = test_pool)

predictions_train = catboost_model.predict(X_train)
predictions_test = catboost_model.predict(X_test)
predictions_validation = catboost_model.predict(X_validation)

feature_importance = catboost_model.get_feature_importance(data=train_pool, type='PredictionValuesChange')
# predictions = pd.DataFrame(predictions)
# predictions.to_csv("D:\Acsq\BIM硕士第三学期\AD\DCA\DCA PFS validation.csv")


def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, sensitivity, specificity, precision, f1_score


# 为训练集、测试集和验证集计算指标
metrics_train = calculate_metrics(y_train_noisy, predictions_train)
metrics_test = calculate_metrics(y_test_noisy, predictions_test)
metrics_validation = calculate_metrics(y_validation_noisy, predictions_validation)

# 将所有指标存储在一个列表中
all_metrics = [metrics_train, metrics_test, metrics_validation]

# 计算平均值和标准差
mean_values = {
    'Accuracy': np.mean([metrics[0] for metrics in all_metrics]),
    'Sensitivity': np.mean([metrics[1] for metrics in all_metrics]),
    'Specificity': np.mean([metrics[2] for metrics in all_metrics]),
    'Precision': np.mean([metrics[3] for metrics in all_metrics]),
    'F1 Score': np.mean([metrics[4] for metrics in all_metrics])
}
std_values = {
    'Accuracy': np.std([metrics[0] for metrics in all_metrics], ddof=1),
    'Sensitivity': np.std([metrics[1] for metrics in all_metrics], ddof=1),
    'Specificity': np.std([metrics[2] for metrics in all_metrics], ddof=1),
    'Precision': np.std([metrics[3] for metrics in all_metrics], ddof=1),
    'F1 Score': np.std([metrics[4] for metrics in all_metrics], ddof=1)
}

# 创建一个DataFrame来展示每个数据集的指标
individual_metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Accuracy': [metrics[0] for metrics in all_metrics],
    'Sensitivity': [metrics[1] for metrics in all_metrics],
    'Specificity': [metrics[2] for metrics in all_metrics],
    'Precision': [metrics[3] for metrics in all_metrics],
    'F1 Score': [metrics[4] for metrics in all_metrics]
})

# 为每个指标创建一个格式化的“平均值 ± 标准差”字符串
mean_std_formatted = pd.Series({
    'Accuracy': f"{mean_values['Accuracy']:.3f} ± {std_values['Accuracy']:.3f}",
    'Sensitivity': f"{mean_values['Sensitivity']:.3f} ± {std_values['Sensitivity']:.3f}",
    'Specificity': f"{mean_values['Specificity']:.3f} ± {std_values['Specificity']:.3f}",
    'Precision': f"{mean_values['Precision']:.3f} ± {std_values['Precision']:.3f}",
    'F1 Score': f"{mean_values['F1 Score']:.3f} ± {std_values['F1 Score']:.3f}"
})

# 打印结果
print("Individual Dataset Metrics:")
print(individual_metrics_df)
print("\nOverall Metrics (Mean ± Std):")
print(mean_std_formatted)
# 获取训练集和测试集的预测概率
train_probs = catboost_model.predict_proba(X_train)[:, 1]
test_probs = catboost_model.predict_proba(X_test)[:, 1]
validation_probs = catboost_model.predict_proba(X_validation)[:,1]
# 计算 ROC 曲线的各项指标
train_fpr, train_tpr, _ = roc_curve(y_train_noisy, train_probs)
test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
validation_fpr,validation_tpr,_ = roc_curve(y_validation,validation_probs)
# 计算 ROC 曲线下面积（AUC）
train_auc = roc_auc_score(y_train_noisy, train_probs)
test_auc = roc_auc_score(y_test, test_probs)
validation_auc = roc_auc_score(y_validation,validation_probs)


# 画出 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f'Train AUC = {train_auc:.3f}')
plt.plot(test_fpr, test_tpr, label=f'Internel Validation AUC = {test_auc:.3f}')
plt.plot(validation_fpr,validation_tpr,label = f'Externel Validation AUC = {validation_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC Curve', fontsize=25)
plt.legend()
plt.show()
# 获取训练过程中的指标值
train_metric_values = catboost_model.get_evals_result()['learn']['Logloss']
test_metric_values = catboost_model.get_evals_result()['validation']['Logloss']
validation_metric_values = catboost_model.get_evals_result()['validation']['Logloss']

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_metric_values, label='Train')
plt.plot(test_metric_values, label='IV')
plt.plot(validation_metric_values,label = 'EV')
plt.xlabel('Number of iterations', fontsize=20)
plt.ylabel('Loss Function', fontsize=20)
plt.title('CatBoost Learning Curve', fontsize=25)
plt.legend()
plt.show()

feature_importance_sorted_indices = np.argsort(feature_importance)[::-1]  # 降序排序
top_ten_indices = feature_importance_sorted_indices[:10]  # 选择前十个
# 提取模型的特征名称
feature_names = train_pool.get_feature_names()
# 选择前十个特征及其重要性
top_ten_features = [feature_names[i] for i in top_ten_indices]
top_ten_importance = [feature_importance[i] for i in top_ten_indices]
# 绘制条形图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.barh(range(len(top_ten_features)), top_ten_importance, color='skyblue', align='center')
plt.yticks(range(len(top_ten_features)), top_ten_features,fontdict={'size': 25})
plt.xlabel('Importance',fontdict={'size': 25})
plt.title('Top 10 Important Features',fontdict={'size': 30})
plt.gca().invert_yaxis()
plt.show()

# 训练随机森林分类器
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# 训练逻辑回归模型
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
# 获取各个模型在测试集上的预测概率
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lr_probs = lr_model.predict_proba(X_test)[:, 1]
# 计算 ROC 曲线的各项指标
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# 计算 ROC 曲线下面积（AUC）
rf_auc = roc_auc_score(y_test, rf_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# # 训练 XGBoost 分类器
# xgb_model = XGBClassifier(enable_categorical=True)
# xgb_model.fit(X_train_xgboost, y_train)

# 训练 KNN 分类器
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
# 训练 SVM 分类器
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
# 获取各个模型在测试集上的预测概率
# xgb_probs = xgb_model.predict_proba(X_test_xgboost)[:, 1]
knn_probs = knn_model.predict_proba(X_test)[:, 1]
svm_probs = svm_model.predict_proba(X_test)[:, 1]
# 计算 ROC 曲线的各项指标
# xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
# 计算 ROC 曲线下面积（AUC）
# xgb_auc = roc_auc_score(y_test, xgb_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
svm_auc = roc_auc_score(y_test, svm_probs)
# # 训练 LightGBM 分类器
# lgbm_model = LGBMClassifier()
# lgbm_model.fit(X_train, y_train)
# 训练 AdaBoost 分类器
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(X_train, y_train)
# 获取各个模型在测试集上的预测概率
# lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
adaboost_probs = adaboost_model.predict_proba(X_test)[:, 1]
# 计算 ROC 曲线的各项指标
# lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_probs)
adaboost_fpr, adaboost_tpr, _ = roc_curve(y_test, adaboost_probs)
# 计算 ROC 曲线下面积（AUC）
# lgbm_auc = roc_auc_score(y_test, lgbm_probs)
adaboost_auc = roc_auc_score(y_test, adaboost_probs)
# 训练随机森林分类器
rf_model.fit(X_train, y_train)
# 训练逻辑回归模型
lr_model.fit(X_train, y_train)

# 获取训练集上的预测概率
rf_train_probs = rf_model.predict_proba(X_train)[:, 1]
lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
# xgb_train_probs = xgb_model.predict_proba(X_train_xgboost)[:,1]
# 计算训练集上的 ROC 曲线的各项指标
rf_train_fpr, rf_train_tpr, _ = roc_curve(y_train, rf_train_probs)
lr_train_fpr, lr_train_tpr, _ = roc_curve(y_train, lr_train_probs)
# xgb_train_fpr, xgb_train_tpr, _ = roc_curve(y_train,xgb_train_probs)
# 计算训练集上的 ROC 曲线下面积（AUC）
rf_train_auc = roc_auc_score(y_train, rf_train_probs)
lr_train_auc = roc_auc_score(y_train, lr_train_probs)
# xgb_train_auc = roc_auc_score(y_train,xgb_train_probs)
# 对于其他模型，按照类似的步骤进行操作
# 训练 KNN 分类器
knn_model.fit(X_train, y_train)
# 训练 SVM 分类器
svm_model.fit(X_train, y_train)
# 训练 AdaBoost 分类器
adaboost_model.fit(X_train, y_train)

# 获取各个模型在训练集上的预测概率
knn_train_probs = knn_model.predict_proba(X_train)[:, 1]
svm_train_probs = svm_model.predict_proba(X_train)[:, 1]
adaboost_train_probs = adaboost_model.predict_proba(X_train)[:, 1]

# 计算训练集上的 ROC 曲线的各项指标
knn_train_fpr, knn_train_tpr, _ = roc_curve(y_train, knn_train_probs)
svm_train_fpr, svm_train_tpr, _ = roc_curve(y_train, svm_train_probs)
adaboost_train_fpr, adaboost_train_tpr, _ = roc_curve(y_train, adaboost_train_probs)

# 计算训练集上的 ROC 曲线下面积（AUC）
knn_train_auc = roc_auc_score(y_train, knn_train_probs)
svm_train_auc = roc_auc_score(y_train, svm_train_probs)
adaboost_train_auc = roc_auc_score(y_train, adaboost_train_probs)
# 画出 ROC 曲线
# 画出训练集的 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f'CatBoost AUC = {train_auc:.3f}')
plt.plot(rf_train_fpr, rf_train_tpr, label=f'Random Forest AUC = {rf_train_auc:.3f}')
plt.plot(lr_train_fpr, lr_train_tpr, label=f'Logistic Regression AUC = {lr_train_auc:.3f}')
# plt.plot(xgb_train_fpr, xgb_train_tpr, label=f'XGBoost AUC = {xgb_train_auc:.3f}')
plt.plot(knn_train_fpr, knn_train_tpr, label=f'KNN AUC = {knn_train_auc:.3f}')
plt.plot(svm_train_fpr, svm_train_tpr, label=f'SVM AUC = {svm_train_auc:.3f}')
# plt.plot(lgbm_train_fpr, lgbm_train_tpr, label=f'LightGBM AUC = {lgbm_train_auc:.3f}')
plt.plot(adaboost_train_fpr, adaboost_train_tpr, label=f'AdaBoost AUC = {adaboost_train_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate',fontdict={'size': 20})
plt.ylabel('True Positive Rate',fontdict={'size': 20})
plt.title('Train ROC Curve Comparison',fontdict={'size': 25})
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(test_fpr, test_tpr, label=f'CatBoost AUC = {test_auc:.3f}')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest AUC = {rf_auc:.3f}')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression AUC = {lr_auc:.3f}')
# plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost AUC = {xgb_auc:.2f}')
plt.plot(knn_fpr, knn_tpr, label=f'KNN AUC = {knn_auc:.3f}')
plt.plot(svm_fpr, svm_tpr, label=f'SVM AUC = {svm_auc:.3f}')
# plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM AUC = {lgbm_auc:.2f}')
plt.plot(adaboost_fpr, adaboost_tpr, label=f'AdaBoost AUC = {adaboost_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate',fontdict={'size': 20})
plt.ylabel('True Positive Rate',fontdict={'size': 20})
plt.title('Internal Validation ROC Curve Comparison',fontdict={'size': 25})
plt.legend()
plt.show()