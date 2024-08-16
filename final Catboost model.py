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





combined_data = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data_OS.csv").dropna()
combined_data_LC = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data LC.csv").dropna()
combined_data_PFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data PFS.csv").dropna()
combined_data_LRFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data LRFS.csv").dropna()

after_data_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New OS.csv").dropna()
after_data_LC = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - LC.csv").dropna()
after_data_PFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - PFS.csv").dropna()
after_data_LRFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - LRFS.csv").dropna()
samples_420 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples.csv").dropna()
moved_samples = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after.csv").dropna()
before_data_420_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before 420 samples OS 7.12.csv").dropna()
before_moved_samples_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before OS 7.12.csv").dropna()
final_data_0S_before = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/before OS 7.19.csv").dropna()
final_data_LRFS = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 LRFS.csv").dropna()

final_data_0S_450 = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_labeled_samples 370.csv").dropna()
final_data_0S_42 = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_unlabeled_samples 21 7.21.csv").dropna()
test_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment test data.csv").dropna()

data_22 = final_data_0S_42.drop(test_data.index)



before_data_LRFS = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment LRFS data.csv").dropna()
train_LRFS_labeled = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.1 LRFS.csv").dropna()
test_LRFS_unlabeled = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.1 LRFS.csv").dropna().drop(['LRFS_m'],axis=1)


before_data_PFS = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment-8.2 PFS.csv").dropna()
before_data_LC = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment-8.2 LC.csv").dropna()

before_data_OS = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 OS.csv").dropna()

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



z_score = stats.zscore(before_data_PFS)
threshold = 4
outliers = (z_score > threshold).any(axis=1)
data_no_outliers =before_data_PFS[~outliers]
data_no_outliers = data_no_outliers.dropna()

print(data_no_outliers)

threshold = 0.8
corr_matrix = data_no_outliers.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

X_final = data_no_outliers.drop(to_drop, axis=1)
outside_validation = X_final.sample(n = 60,random_state=42)
#outside_validation.to_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 PFS external validation.csv")
print(outside_validation)
# X_drop = data_no_outliers.drop(X_final,axis=1)
# X_drop.to_csv("D:\Acsq\BIM硕士第三学期\AD/data drop.csv")
# X_final.to_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 OS after preprocessing.csv")


X = X_final.drop(['PFS_m','PFS'],axis=1).astype(str)
y = X_final['PFS']


final_data_0S_42.astype(str)

X_moved_samples = final_data_0S_42.drop(['OS','OS_m'],axis=1)
y_moved_samples = final_data_0S_42['OS']


X_test_new = test_data.drop(['OS','OS_m'],axis=1).astype(str)
y_test_new = test_data['OS']

scaler =StandardScaler()
keywords = ['treatment', 'TL', 'original']
columns_to_standardize = [col for col in X.columns if any(keyword in col for keyword in keywords)]
X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
#X_test_new[columns_to_standardize] = scaler.fit_transform(X_test_new[columns_to_standardize])

print(X.to_string())
X_final.astype(str)

X_train, X_test,y_train,y_test = train_test_split(X , y ,test_size=0.3, random_state=42)
#y_train_noisy = add_noise_to_labels(y_train, noise_level=0.08)

train_pool = Pool(X_train, y_train, cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])
test_pool = Pool(X_test, y_test, cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])


catboost_model = CatBoostClassifier(loss_function = LoglossObjective(),eval_metric = 'Logloss', cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])

catboost_model.fit(train_pool,eval_set = test_pool)

feature_importance = catboost_model.get_feature_importance(data=train_pool, type='PredictionValuesChange')

predictions = catboost_model.predict(X_test)
predictions=pd.DataFrame(predictions)



# print(predictions)





# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = conf_matrix.ravel()
# 计算准确率、敏感度、特异度
accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
print("模型准确率：", accuracy)
print("敏感度：", sensitivity)
print("特异度：", specificity)
print("精确率：", precision)
print("F1分数：", f1_score)
# 获取训练集和测试集的预测概率
train_probs = catboost_model.predict_proba(X_train)[:, 1]
test_probs = catboost_model.predict_proba(X_test)[:, 1]
# 计算 ROC 曲线的各项指标
train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
# 计算 ROC 曲线下面积（AUC）
train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)



# 画出 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f'Train AUC = {train_auc:.3f}')
plt.plot(test_fpr, test_tpr, label=f'Internal Validation AUC = {test_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC Curve', fontsize=25)
plt.legend()
plt.show()
# 获取训练过程中的指标值
train_metric_values = catboost_model.get_evals_result()['learn']['Logloss']
test_metric_values = catboost_model.get_evals_result()['validation']['Logloss']

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_metric_values, label='Train')
plt.plot(test_metric_values, label='Internal Validation')
plt.xlabel('Number of iterations', fontsize=20)
plt.ylabel('Loss Function', fontsize=20)
plt.title('CatBoost Learning Curve', fontsize=25)
plt.legend()
plt.show()
# 使用Kaplan-Meier估计生存函数

kmf = KaplanMeierFitter()
durations_OS = before_data_OS['OS_m']
events_OS = before_data_OS['OS']
durations_LRFS = before_data_PFS['PFS_m']
events_PFS = before_data_PFS['PFS']
kmf1 = KaplanMeierFitter()
kmf1.fit(durations_OS, event_observed=events_OS)
kmf2 = KaplanMeierFitter()
kmf2.fit(durations_LRFS, event_observed=events_PFS)
fig, ax = plt.subplots(figsize=(8, 6))
kmf1.plot(ax=ax, label='OS')
kmf2.plot(ax=ax, label='PFS')
plt.xlabel('Time',fontsize=27)
plt.ylabel('Survival probability',fontsize=27)
plt.title('Kaplan-Meier Curves of OS and PFS',fontsize=27)
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
plt.yticks(range(len(top_ten_features)), top_ten_features,fontdict={'size': 20})
plt.xlabel('Importance',fontdict={'size': 25})
plt.title('Top 10 Important Features',fontdict={'size': 25})
plt.gca().invert_yaxis()
plt.show()




#调参
def catboost_parameter_search(X_train, y_train, X_test, y_test, param_grid):
    evaluation_metrics = []
    roc_auc_values = []

    # Grid search over parameter grid
    for depth in param_grid['depth']:
        for lr in param_grid['learning_rate']:
            for iterations in param_grid['iterations']:
                # for penalty in param_grid['penalty']:
                    # for reward_factor in param_grid['reward_factor']:
                        catboost_model = CatBoostClassifier(
                            depth=depth,
                            learning_rate=lr,
                            iterations=iterations,
                            loss_function=LoglossObjective(),
                            eval_metric='Logloss',
                            early_stopping_rounds=50
                        )

                        train_pool = Pool(X_train, y_train,cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])
                        test_pool = Pool(X_test, y_test, cat_features=['Age','Location', 'N', 'TNM','PTV_Dose', 'GTV_Dose','ECOG','T','Chemotherapy'])
                        catboost_model.fit(train_pool, eval_set=test_pool)

                        # Calculate train AUC
                        y_prob_train = catboost_model.predict_proba(X_train)[:, 1]
                        fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
                        roc_auc_train = auc(fpr_train, tpr_train)

                        # Calculate test AUC
                        y_prob_test = catboost_model.predict_proba(X_test)[:, 1]
                        fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
                        roc_auc_test = auc(fpr_test, tpr_test)
                        threshold = 0.5
                        y_pred_test = (y_prob_test > threshold).astype(int)
                        confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
                        tn, fp, fn, tp = confusion_matrix_test.ravel()

                        sensitivity = tp / (tp + fn)
                        specificity = tn / (tn + fp)
                        ppv = tp / (tp + fp)
                        npv = tn / (tn + fn)
                        icc = (tp + tn) / (tp + tn + fp + fn)
                        auc_ci = np.nan_to_num(roc_auc_test)

                        auc_ci_lower, auc_ci_upper = stats.norm.interval(0.95, loc=auc_ci,
                                                                             scale=np.sqrt(auc_ci * (1 - auc_ci) / len(y_test)))

                        evaluation_metrics.append(
                                {'Depth': depth, 'Learning Rate': lr, 'Iterations': iterations, 'AUC': auc_ci, 'PPV': ppv,
                                 'NPV': npv, 'ICC': icc, 'Sensitivity': sensitivity, 'Specificity': specificity,
                                 '95% CI Lower': auc_ci_lower, '95% CI Upper': auc_ci_upper})

                        roc_auc_values.append((depth, lr, iterations, roc_auc_test))



                # Print the DataFrame with all evaluation metrics in tabular format

                        # Store evaluation metrics
                        evaluation_metrics.append({
                            'Depth': depth,
                            'Learning Rate': lr,
                            'Iterations': iterations,
                            # 'Penalty': penalty,
                            # 'Reward Factor': reward_factor,
                            'Test AUC': roc_auc_test,
                            'Train AUC': roc_auc_train
                        })

    # Create DataFrame for evaluation metrics
    evaluation_metrics_df = pd.DataFrame(evaluation_metrics)

    # Sort roc_auc_values by test AUC and select top 10 combinations
    roc_auc_values = sorted(roc_auc_values,key=lambda x: x[3] ,reverse=True)[:5]

    evaluation_metrics_df = evaluation_metrics_df.sort_values(by='AUC', ascending=False)
    print(tabulate(evaluation_metrics_df, headers='keys', tablefmt='psql'))


    # Visualize ROC curves for top 10 combinations on train set
    plt.figure(figsize=(12, 8))
    for depth, lr, iterations, roc_auc_test in roc_auc_values:
        catboost_model = CatBoostClassifier(
            depth=depth,
            learning_rate=lr,
            iterations=iterations
        )
        catboost_model.fit(X_train, y_train)

        y_prob_train = catboost_model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
        roc_auc_train_current = auc(fpr_train, tpr_train)

        plt.plot(fpr_train, tpr_train, lw=2,
                 label=f'Depth: {depth}, LR: {lr}, Iter: {iterations}, \n(train AUC = {roc_auc_train_current:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC Curves for Top 5 Parameter Combinations on Train Set', fontsize=20)
    plt.legend(loc='lower right', fontsize='large')
    plt.show()

    # Visualize ROC curves for top 10 combinations on test set
    plt.figure(figsize=(12, 8))
    for depth, lr, iterations,roc_auc_test in roc_auc_values:
        catboost_model = CatBoostClassifier(depth=depth, learning_rate=lr, iterations=iterations)
        catboost_model.fit(X_train, y_train)
        y_prob_test = catboost_model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
        plt.plot(fpr_test, tpr_test, lw=2,
                     label=f'Depth: {depth}, LR: {lr}, Iter: {iterations} (Internal Validation AUC = {roc_auc_test:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC Curves for Top 5 Parameter Combinations on Internal Validation Set', fontsize=20)
    plt.legend(loc='lower right', fontsize='large')
    plt.show()

# 调用函数进行网格搜索
param_grid = {
    'depth': [ 4, 6,8],
    'learning_rate': [0.01, 0.03,0.1],
    'iterations': [400, 600,800,1000],

    # 'reward_factor':[1.2,1.5,2]
}

catboost_parameter_search(X_train, y_train, X_test, y_test, param_grid)

# # 创建数据副本
# X_train_xgboost = X_train.copy()
# X_test_xgboost = X_test.copy()
#
#
# # 创建 LabelEncoder 对象
# label_encoder = preprocessing.LabelEncoder()
#
# # 对每一列进行迭代，将对象类型的列进行编码转换
# for column in X_train_xgboost:
#     if X_train_xgboost[column].dtype.kind == 'O':
#         X_train_xgboost[column] = label_encoder.fit_transform(X_train_xgboost[column])
#         X_test_xgboost[column] = label_encoder.transform(X_test_xgboost[column])


#比较
#训练随机森林分类器
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# # 训练逻辑回归模型
# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(X_train, y_train)
# # 获取各个模型在测试集上的预测概率
# rf_probs = rf_model.predict_proba(X_test)[:, 1]
# lr_probs = lr_model.predict_proba(X_test)[:, 1]
# # 计算 ROC 曲线的各项指标
# rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
# lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# # 计算 ROC 曲线下面积（AUC）
# rf_auc = roc_auc_score(y_test, rf_probs)
# lr_auc = roc_auc_score(y_test, lr_probs)
# # # 训练 XGBoost 分类器
# # xgb_model = XGBClassifier(enable_categorical=True)
# # xgb_model.fit(X_train_xgboost, y_train)
#
# # 训练 KNN 分类器
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train, y_train)
# # 训练 SVM 分类器
# svm_model = SVC(probability=True)
# svm_model.fit(X_train, y_train)
# # 获取各个模型在测试集上的预测概率
# # xgb_probs = xgb_model.predict_proba(X_test_xgboost)[:, 1]
# knn_probs = knn_model.predict_proba(X_test)[:, 1]
# svm_probs = svm_model.predict_proba(X_test)[:, 1]
# # 计算 ROC 曲线的各项指标
# # xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
# knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
# svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
# # 计算 ROC 曲线下面积（AUC）
# # xgb_auc = roc_auc_score(y_test, xgb_probs)
# knn_auc = roc_auc_score(y_test, knn_probs)
# svm_auc = roc_auc_score(y_test, svm_probs)
# # # 训练 LightGBM 分类器
# # lgbm_model = LGBMClassifier()
# # lgbm_model.fit(X_train, y_train)
# # 训练 AdaBoost 分类器
# adaboost_model = AdaBoostClassifier()
# adaboost_model.fit(X_train, y_train)
# # 获取各个模型在测试集上的预测概率
# # lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
# adaboost_probs = adaboost_model.predict_proba(X_test)[:, 1]
# # 计算 ROC 曲线的各项指标
# # lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_probs)
# adaboost_fpr, adaboost_tpr, _ = roc_curve(y_test, adaboost_probs)
# # 计算 ROC 曲线下面积（AUC）
# # lgbm_auc = roc_auc_score(y_test, lgbm_probs)
# adaboost_auc = roc_auc_score(y_test, adaboost_probs)
# # 训练随机森林分类器
# rf_model.fit(X_train, y_train)
# # 训练逻辑回归模型
# lr_model.fit(X_train, y_train)
#
# # 获取训练集上的预测概率
# rf_train_probs = rf_model.predict_proba(X_train)[:, 1]
# lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
# # xgb_train_probs = xgb_model.predict_proba(X_train_xgboost)[:,1]
# # 计算训练集上的 ROC 曲线的各项指标
# rf_train_fpr, rf_train_tpr, _ = roc_curve(y_train, rf_train_probs)
# lr_train_fpr, lr_train_tpr, _ = roc_curve(y_train, lr_train_probs)
# # xgb_train_fpr, xgb_train_tpr, _ = roc_curve(y_train,xgb_train_probs)
# # 计算训练集上的 ROC 曲线下面积（AUC）
# rf_train_auc = roc_auc_score(y_train, rf_train_probs)
# lr_train_auc = roc_auc_score(y_train, lr_train_probs)
# # xgb_train_auc = roc_auc_score(y_train,xgb_train_probs)
# # 对于其他模型，按照类似的步骤进行操作
# # 训练 KNN 分类器
# knn_model.fit(X_train, y_train)
# # 训练 SVM 分类器
# svm_model.fit(X_train, y_train)
# # 训练 AdaBoost 分类器
# adaboost_model.fit(X_train, y_train)
#
# # 获取各个模型在训练集上的预测概率
# knn_train_probs = knn_model.predict_proba(X_train)[:, 1]
# svm_train_probs = svm_model.predict_proba(X_train)[:, 1]
# adaboost_train_probs = adaboost_model.predict_proba(X_train)[:, 1]
#
# # 计算训练集上的 ROC 曲线的各项指标
# knn_train_fpr, knn_train_tpr, _ = roc_curve(y_train, knn_train_probs)
# svm_train_fpr, svm_train_tpr, _ = roc_curve(y_train, svm_train_probs)
# adaboost_train_fpr, adaboost_train_tpr, _ = roc_curve(y_train, adaboost_train_probs)
#
# # 计算训练集上的 ROC 曲线下面积（AUC）
# knn_train_auc = roc_auc_score(y_train, knn_train_probs)
# svm_train_auc = roc_auc_score(y_train, svm_train_probs)
# adaboost_train_auc = roc_auc_score(y_train, adaboost_train_probs)
# # 画出 ROC 曲线
# # 画出训练集的 ROC 曲线
# plt.figure(figsize=(8, 6))
# plt.plot(train_fpr, train_tpr, label=f'CatBoost AUC = {train_auc:.3f}')
# plt.plot(rf_train_fpr, rf_train_tpr, label=f'Random Forest AUC = {rf_train_auc:.3f}')
# plt.plot(lr_train_fpr, lr_train_tpr, label=f'Logistic Regression AUC = {lr_train_auc:.3f}')
# # plt.plot(xgb_train_fpr, xgb_train_tpr, label=f'XGBoost AUC = {xgb_train_auc:.3f}')
# plt.plot(knn_train_fpr, knn_train_tpr, label=f'KNN AUC = {knn_train_auc:.3f}')
# plt.plot(svm_train_fpr, svm_train_tpr, label=f'SVM AUC = {svm_train_auc:.3f}')
# # plt.plot(lgbm_train_fpr, lgbm_train_tpr, label=f'LightGBM AUC = {lgbm_train_auc:.3f}')
# plt.plot(adaboost_train_fpr, adaboost_train_tpr, label=f'AdaBoost AUC = {adaboost_train_auc:.3f}')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate',fontdict={'size': 20})
# plt.ylabel('True Positive Rate',fontdict={'size': 20})
# plt.title('Train ROC Curve Comparison',fontdict={'size': 25})
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.plot(test_fpr, test_tpr, label=f'CatBoost AUC = {test_auc:.3f}')
# plt.plot(rf_fpr, rf_tpr, label=f'Random Forest AUC = {rf_auc:.3f}')
# plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression AUC = {lr_auc:.3f}')
# # plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost AUC = {xgb_auc:.2f}')
# plt.plot(knn_fpr, knn_tpr, label=f'KNN AUC = {knn_auc:.3f}')
# plt.plot(svm_fpr, svm_tpr, label=f'SVM AUC = {svm_auc:.3f}')
# # plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM AUC = {lgbm_auc:.2f}')
# plt.plot(adaboost_fpr, adaboost_tpr, label=f'AdaBoost AUC = {adaboost_auc:.3f}')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate',fontdict={'size': 20})
# plt.ylabel('True Positive Rate',fontdict={'size': 20})
# plt.title('Test ROC Curve Comparison',fontdict={'size': 25})
# plt.legend()
# plt.show()


#调参
