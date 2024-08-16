import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import Pool
from catboost import CatBoost
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, RFECV, RFE
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import Lasso
import itertools
from sklearn.svm import SVC
from forestplot import forestplot

sample_labeled_ig = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_ig.csv").dropna()
sample_labeled = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_us.csv").dropna()
sample_labeled_bald = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_bald.csv").dropna()
sample_labeled_bald_LC = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_bald_LC.csv").dropna()
sample_labeled_bald_PFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_bald_PFS.csv").dropna()
sample_labeled_bald_LRFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\sample_labeld_bald_LRFS.csv").dropna()
after_sample_labeled_bald_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/after_data_bald.csv").dropna()
after_sample_labeled_bald_OS_64 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/after_data_bald6.4.csv").dropna()
before_sample_labled_bald_OS_64 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before_data_bald6.4.csv").dropna()
before_and_after_labled_bald_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after data_bald6.5.csv").dropna()
all_samples = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/all samples.csv").dropna()
removed_sampled = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/removed.csv").dropna()
before_420 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\labeled 420 before.csv").dropna()
removed_22 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22.csv").dropna()
before_and_after_labeled_420 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples.csv").dropna()
before_and_after_moved_22 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after.csv").dropna()
before_and_after_labeled_350 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 350 samples.csv").dropna()
before_and_after_moved_92 = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 92 before and after.csv").dropna()
# without_handle_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\combined data_OS.csv").dropna()
before_and_after_sample_labled_bald_LC = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples LC.csv").dropna()
removed_22_LC = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after LC.csv").dropna()
before_and_after_sample_labled_bald_PFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples PFS.csv").dropna()
removed_22_PFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after PFS.csv").dropna()
before_and_after_sample_labled_bald_LRFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before and after 420 samples LRFS.csv").dropna()
removed_22_LRFS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before and after LRFS.csv").dropna()
before_OS = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\EC_after_Treatment_New - OS.csv").dropna()
before_OS_420_new = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master/before 420 samples OS.csv").dropna()
removed_22_OS_new = pd.read_csv("D:\Acsq\BIM硕士第二学期\w\pycox-master\move 22 before OS.csv").dropna()
final_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment 7.12 OS after preprocessing.csv").dropna()



final_450_samples = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment_labeled_samples 370 7.21.csv").dropna()
final_42_samples = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 OS external validation.csv").dropna()
drop_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\EC_before_Treatment_unlabeled_samples 22 7.22.csv").dropna()
print(final_42_samples.to_string())


new_data = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/before OS 7.19.csv").dropna()
new_data_with_exp = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\data 7.24\EC_before_Treatment with Exponential 7.24.csv").dropna()
new_data_with_log = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD\data 7.24\EC_before_Treatment with Log.csv").dropna()


new_LRFS_train = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.1 LRFS.csv").dropna()
new_LRFS_test = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.1 LRFS.csv").dropna()

new_PFS_train = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.2 PFS.csv").dropna()
new_PFS_test = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment 7.12 PFS external validation.csv").dropna()


new_LC_train = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment labeled 8.2 LC.csv").dropna()
new_LC_test = pd.read_csv("D:\Acsq\BIM硕士第三学期\AD/final data\EC_before_Treatment test data 8.2 LC.csv").dropna()



new_X = new_data.drop(['OS','OS_m'],axis=1)
print(new_X)
new_y = new_data['OS']
time = new_data['OS_m']
new_train_X,new_test_X,new_y_train,new_y_test = train_test_split(new_X,new_y,test_size=0.3,random_state=42)




# 应用Lasso回归
lasso = Lasso(alpha=1)  # alpha参数需要根据实际情况调优
lasso.fit(new_X,new_y)
# 获取特征的重要性
coef = lasso.coef_
selected_features = new_X.columns[lasso.coef_ != 0]
selected_features = selected_features.tolist()
selected_data = new_X[selected_features]

new_data_after_lasso = pd.concat([selected_data,new_y,time],axis=1)


# #SVM
# svm = SVC(kernel='linear', C=1)
# svm.fit(new_train_X, new_y_train)
# coef_SVM = SVC.coef_
# # 获取特征重要性，筛选出具有非零权重的特征
# important_features_SVM = new_X.columns[SVC.coef_ != 0]
# selected_features_SVM = important_features_SVM.tolist()
# selected_data_SVM = new_X[selected_features_SVM]
# print(selected_features_SVM)

# #Catboost
#
# #y_train_noisy = add_noise_to_labels(y_train, noise_level=0.08)
# train_pool = Pool(new_train_X, new_y_train)
# test_pool = Pool(new_test_X, new_y_train)
# catboost_model = CatBoostClassifier(depth=8, iterations=400, learning_rate=0.1,early_stopping_rounds = 50,loss_function = 'Logloss',eval_metric = 'Logloss')
# catboost_model.fit(train_pool,eval_set = test_pool)
# feature_importance = catboost_model.get_feature_importance(data=train_pool, type='PredictionValuesChange')
# print(feature_importance)




X_train,X_test = train_test_split(all_samples,test_size=0.3,random_state=42)
X_train_cox_ig, X_test_cox_ig = train_test_split(sample_labeled_ig, test_size=0.3,random_state=42)
X_train_cox_us, X_test_cox_us = train_test_split(sample_labeled, test_size=0.3,random_state=42)
X_train_cox_bald, X_test_cox_bald = train_test_split(before_420, test_size=0.3,random_state=42)
X_train_cox_bald_LC, X_test_cox_bald_LC = train_test_split(sample_labeled_bald_LC, test_size=0.3,random_state=42)
X_train_cox_bald_PFS, X_test_cox_bald_PFS = train_test_split(sample_labeled_bald_PFS, test_size=0.3,random_state=42)
X_train_cox_bald_LRFS, X_test_cox_bald_LRFS = train_test_split(sample_labeled_bald_LRFS, test_size=0.3,random_state=42)
X_train_cox_bald_after_OS, X_test_cox_bald_after_OS= train_test_split(after_sample_labeled_bald_OS, test_size=0.3,random_state=42)
X_train_cox_bald_after_OS_64, X_test_cox_bald_after_OS_64= train_test_split(after_sample_labeled_bald_OS_64, test_size=0.3,random_state=42)
X_train_cox_bald_before_64, X_test_cox_bald_before_64 = train_test_split(before_sample_labled_bald_OS_64, test_size=0.3,random_state=42)
X_train_cox_bald_before_and_after,X_test_cox_bald_before_and_after = train_test_split(before_and_after_labeled_420, test_size=0.35,random_state=42)
# X_train_cox_without_handle, X_test_cox_without_handle = train_test_split(without_handle_OS, test_size=0.3,random_state=42)
X_train_cox_bald_LC_420, X_test_cox_bald_LC_420 = train_test_split(before_and_after_sample_labled_bald_LC , test_size=0.3,random_state=42)
X_train_cox_bald_PFS_420,X_test_cox_bald_PFS_420 = train_test_split(before_and_after_sample_labled_bald_PFS,test_size=0.3,random_state=42)
X_train_cox_bald_LRFS_420,X_test_cox_bald_LRFS_420 = train_test_split(before_and_after_sample_labled_bald_LRFS,test_size=0.3,random_state=42)
X_train_cox_before_OS_420, X_test_cox_before_OS_420= train_test_split(before_OS_420_new, test_size=0.3,random_state=42)
X_train_final,X_test_final = train_test_split(final_data,test_size=0.3,random_state=42)
X_train_final_450,X_test_final_450 = train_test_split(final_450_samples,test_size=0.3,random_state=42)





X_train_cox = pd.DataFrame(X_train,columns=all_samples.columns)
X_test_cox = pd.DataFrame(X_test,columns=all_samples.columns)


X_train_cox_ig = pd.DataFrame(X_train_cox_ig,columns=sample_labeled_ig.columns)
X_test_cox_ig = pd.DataFrame(X_test_cox_ig,columns=sample_labeled_ig.columns)

X_train_cox_us = pd.DataFrame(X_train_cox_us,columns=sample_labeled.columns)
X_test_cox_us = pd.DataFrame(X_test_cox_us,columns=sample_labeled.columns)

X_train_cox_bald = pd.DataFrame(X_train_cox_bald,columns=before_420.columns)
X_test_cox_bald = pd.DataFrame(X_test_cox_bald,columns=before_420.columns)
#X_test_cox_bald = pd.concat([X_test_cox_bald,removed_22],axis=0)

X_train_cox_bald_LC = pd.DataFrame(X_train_cox_bald_LC,columns=sample_labeled_bald_LC.columns)
X_test_cox_bald_LC = pd.DataFrame(X_test_cox_bald_LC,columns=sample_labeled_bald_LC.columns)

X_train_cox_bald_PFS = pd.DataFrame(X_train_cox_bald_PFS,columns=sample_labeled_bald_PFS.columns)
X_test_cox_bald_PFS = pd.DataFrame(X_test_cox_bald_PFS,columns=sample_labeled_bald_PFS.columns)

X_train_cox_bald_LRFS = pd.DataFrame(X_train_cox_bald_LRFS,columns=sample_labeled_bald_LRFS.columns)
X_test_cox_bald_LRFS = pd.DataFrame(X_test_cox_bald_LRFS,columns=sample_labeled_bald_LRFS.columns)

X_train_cox_bald_after_OS = pd.DataFrame(X_train_cox_bald_after_OS,columns=after_sample_labeled_bald_OS.columns)
X_test_cox_bald_after_OS = pd.DataFrame(X_test_cox_bald_after_OS,columns=after_sample_labeled_bald_OS.columns)

X_train_cox_bald_after_OS_64 = pd.DataFrame(X_train_cox_bald_after_OS_64,columns=after_sample_labeled_bald_OS_64.columns)
X_test_cox_bald_after_OS_64 = pd.DataFrame(X_test_cox_bald_after_OS_64,columns=after_sample_labeled_bald_OS_64.columns)

X_train_cox_bald_before_64 = pd.DataFrame(X_train_cox_bald_before_64,columns=before_sample_labled_bald_OS_64.columns)
X_test_cox_bald_before_64 = pd.DataFrame(X_test_cox_bald_before_64,columns=before_sample_labled_bald_OS_64.columns)


X_train_cox_bald_before_and_after = pd.DataFrame(X_train_cox_bald_before_and_after,columns=before_and_after_labled_bald_OS.columns)
X_test_cox_bald_before_and_after = pd.DataFrame(X_test_cox_bald_before_and_after,columns=before_and_after_labled_bald_OS.columns)
X_test_cox_bald_before_and_after = pd.concat([X_test_cox_bald_before_and_after,before_and_after_moved_22],axis=0)

# X_train_cox_without_handle = pd.DataFrame(without_handle_OS,columns=without_handle_OS.columns)
# X_test_cox_without_handle = pd.DataFrame(without_handle_OS,columns=without_handle_OS.columns)

X_train_cox_bald_LC_420 = pd.DataFrame(X_train_cox_bald_LC_420,columns=before_and_after_sample_labled_bald_LC .columns)
X_test_cox_bald_LC_420 = pd.DataFrame(X_test_cox_bald_LC_420,columns=before_and_after_sample_labled_bald_LC .columns)
removed_22_LC.astype(str)
removed_22_LC = pd.DataFrame(removed_22_LC,columns=before_and_after_sample_labled_bald_LC.columns)
X_test_cox_bald_LC_420 = pd.concat([removed_22_LC,X_test_cox_bald_LC_420],axis=0)


X_train_cox_bald_PFS_420 = pd.DataFrame(X_train_cox_bald_PFS_420,columns=before_and_after_sample_labled_bald_PFS.columns)
X_test_cox_bald_PFS_420 = pd.DataFrame(X_test_cox_bald_PFS_420,columns=before_and_after_sample_labled_bald_PFS.columns)
X_test_cox_bald_PFS_420 = pd.concat([X_test_cox_bald_PFS_420,removed_22_PFS],axis=0)

X_train_cox_bald_LRFS_420 = pd.DataFrame(X_train_cox_bald_LRFS_420,columns=before_and_after_sample_labled_bald_LRFS.columns)
X_test_cox_bald_LRFS_420 = pd.DataFrame(X_test_cox_bald_LRFS_420,columns=before_and_after_sample_labled_bald_LRFS.columns)
X_test_cox_bald_LRFS_420 = pd.concat([X_test_cox_bald_LRFS_420,removed_22_LRFS],axis=0)

X_train_cox_before_OS_420 = pd.DataFrame(before_OS_420_new,columns=before_OS_420_new.columns)
X_test_cox_before_OS_420 = pd.DataFrame(before_OS_420_new,columns=before_OS_420_new.columns)
X_test_cox_before_OS_420 = pd.concat([before_OS_420_new,removed_22_OS_new],axis=0)

X_train_final = pd.DataFrame(X_train_final,columns=final_data.columns)
X_test_final = pd.DataFrame(X_test_final,columns=final_data.columns)


X_train_final_450 = pd.DataFrame(X_train_final_450,columns=final_450_samples.columns)
X_test_final_450 = pd.DataFrame(X_test_final_450,columns=final_450_samples.columns)
final_42_samples = pd.DataFrame(final_42_samples,columns=final_42_samples.columns)


X_train_370 = final_450_samples
X_train_22 = drop_data
X_test_99 = final_42_samples
weights_370 = np.ones(len(X_train_370)) * 3
weights_22 = np.ones(len(X_train_22)) * 2
weights_99 = np.ones(len(X_test_99)) * 1
X_train_combined = pd.concat([X_train_370,X_train_22],axis=0)
weights_combined = np.concatenate([weights_370, weights_22])
X_train_combined['weights'] = weights_combined

X_after_lasso_train,X_after_lasso_test = train_test_split(new_data_after_lasso,test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train_scaled_lasso = scaler.fit_transform(X_after_lasso_train)
X_test_scaled_lasso = scaler.transform(X_after_lasso_test)


def train_and_evaluate_cox_model(X_train_cox, X_test_cox):
    # 实例化 CoxPHFitter 对象
    cph = CoxPHFitter()
    # 使用训练集 X_train_cox 进行训练

    cph.fit(X_train_cox, duration_col='PFS_m', event_col='PFS')
    # 在测试集上进行预测
    predictions = cph.predict_expectation(X_test_cox)
    # 计算 Concordance Index
    c_index = concordance_index(X_test_cox['PFS_m'], predictions, X_test_cox['PFS'])
    print("Concordance Index on test set: {:.3f}".format(c_index))
    # 显示模型摘要信息
    print("\nTraining Set Model Information:")
    cph.print_summary()

    # 提取特征系数
    feature_importances = cph.params_

    # 排序并选择排名前十的特征
    top_features = feature_importances.abs().sort_values(ascending=False).head(10)

    # 绘制排名前十特征的系数条形图
    plt.figure(figsize=(12, 8))
    top_features.plot(kind='barh')
    plt.title('Top 10 Features by Importance',fontdict={'size': 27})
    plt.xlabel('Coefficient Value',fontdict={'size': 27})
    plt.ylabel('Features',fontdict={'size': 27})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.gca().tick_params(axis='y', labelsize=25)
    plt.show()
    # 绘制模型曲线
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Cox Proportional Hazards Model')
    plt.xlabel('Log-HR')
    plt.ylabel('Survival Function')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['Cox Model'], loc='best')
    plt.show()


train_and_evaluate_cox_model(new_PFS_train,new_PFS_test)





# 假设 X 是包含所有特征的数据框，n_features 是特征总数
# 确保 'OS_m' 和 'OS' 列在数据框中
assert 'PFS_m' in new_PFS_train.columns and 'PFS_m' in new_PFS_test.columns
assert 'PFS' in new_PFS_train.columns and 'PFS' in new_PFS_test.columns

# 获取所有特征的列名，排除 'OS_m' 和 'OS'
all_features = new_PFS_train.columns.tolist()
essential_features = ['PFS_m', 'PFS']
other_features = [f for f in all_features if f not in essential_features]

# 存储 Concordance Index 值
c_index_values = []

# 初始化最大 C 索引和最佳特征组合
max_c_index = 0
best_feature_combination = []

# 循环不同数量的特征
for i in range(1, len(other_features) + 1):
    # 随机选择 i 个其他特征
    selected_features = essential_features + np.random.choice(other_features, i, replace=False).tolist()

    # 训练 Cox 模型
    cph = CoxPHFitter()
    cph.fit(new_PFS_train[selected_features], duration_col='PFS_m', event_col='PFS')

    # 计算 Concordance Index
    predictions = cph.predict_expectation(new_PFS_train[selected_features])
    c_index = concordance_index(new_PFS_train['PFS_m'], predictions, new_PFS_train ['PFS'])
    c_index_values.append(c_index)

    # 记录最大的 C 索引和对应的特征组合
    if c_index > max_c_index:
        max_c_index = c_index
        best_feature_combination = selected_features

    # 打印当前 Concordance Index
    print(f"Number of features: {i}, Concordance Index: {c_index:.3f}")

# 打印C索引最大的特征组合和对应的C索引值
print(f"The best feature combination with the highest C-index is: {best_feature_combination}")
print(f"The highest C-index value is: {max_c_index:.3f}")

# 绘制 Concordance Index 与特征数量的关系图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(other_features) + 1), c_index_values, marker='o')
plt.title('Concordance Index vs Number of Features',fontsize=20)
plt.xlabel('Number of Features',fontsize=20)
plt.ylabel('Concordance Index',fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 提取最佳特征组合
best_features = best_feature_combination

# 再次训练Cox模型，使用找到的最佳特征组合
cph_best = CoxPHFitter()
cph_best.fit(new_PFS_train[best_features], duration_col='PFS_m', event_col='PFS')
# 输出模型结果
print(cph_best.summary.to_string())
# 计算在测试集上的预测
predictions_best = cph_best.predict_expectation(new_PFS_test[best_features])

new_c_index = concordance_index(new_PFS_test['PFS_m'], predictions_best, new_PFS_test['PFS'])
print(f"The new Concordance Index with the best feature combination is: {new_c_index:.3f}")
# 绘制生存曲线 (构造生存函数)
cph_best.plot()
plt.title('Survival Function for Best Features')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()
# 获取除了 'OS' 和 'OS_m' 之外的所有特征
# other_features = [col for col in final_450_samples.columns if col not in ['OS_m', 'OS']]

# # 计算 Concordance Index 值的列表
# c_index_values = []
#
# # 生成所有可能的特征组合，包括 'OS' 和 'OS_m'
# for i in range(1, len(other_features) + 1):
#     for combo in itertools.combinations(other_features, i):
#         # 将 'OS' 和 'OS_m' 加入当前特征组合
#         feature_set = list(combo) + ['OS_m', 'OS']
#
#         # 训练 Cox 模型
#         cph = CoxPHFitter()
#         cph.fit(final_450_samples[feature_set], duration_col='OS_m', event_col='OS')
#
#         # 在测试集上进行预测
#         predictions = cph.predict_expectation(final_42_samples[feature_set])
#         # 计算 Concordance Index
#         c_index = concordance_index(final_42_samples['OS_m'], predictions, final_42_samples['OS'])
#
#         # 存储 Concordance Index 值
#         c_index_values.append((i, c_index))
#
# # 将结果转换为 DataFrame 以便分析
# results_df = pd.DataFrame(c_index_values, columns=['Number of Additional Features', 'Concordance Index'])
#
# # 绘制 Concordance Index 与额外特征数量的关系图
# plt.figure(figsize=(12, 8))
# plt.plot(results_df['Number of Additional Features'], results_df['Concordance Index'], marker='o')
# plt.title('Concordance Index vs Number of Additional Features')
# plt.xlabel('Number of Additional Features')
# plt.ylabel('Concordance Index')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()