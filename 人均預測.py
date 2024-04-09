# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 06:01:27 2024

@author: Dean
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from util import get_dummies, detect_str_columns,model,profit_linechart,profit_linechart_all,logistic_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
# ----設定繪圖-------
import matplotlib.pyplot as plt
import seaborn as sns 

sns.set(rc={'figure.figsize':(15,10)})
import numpy as np

from ucimlrepo import fetch_ucirepo 


  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# 分析缺失值
#y.to_excel('y.xls')


data = pd.concat([X,y], axis=1)


# 將?替换成NA
data.replace('?', pd.NA, inplace=True)

# 删除包含 NaN 值的行
data.dropna(inplace=True)

# income轉換成布林值

data['income'] = data['income'].map({'<=50K': 0, '>50K': 1, '>50K.': 1, '<=50K.': 0})


# one hot encode
str_columns = detect_str_columns(data)
dataset = get_dummies(str_columns, data)

# 特徵工程 排除相關係數低於75%特徵
all_analytics = dataset

#all_analytics.to_excel('分析2.xls')

all_corr = all_analytics.corr()

incom_corr = abs(all_corr['income'])

drop75pct = incom_corr.sort_index().tail(int(len(incom_corr) * 0.75)).index

df = all_analytics.drop(columns=drop75pct)
df = df.drop(columns=['education_Doctorate','education_Preschool','fnlwgt'])

# 訓練前切割
#df.columns
X =df.drop(columns=['income'])
y =df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# XGBOOST
# 命名模型物件
xgb_model = XGBClassifier(n_estimators = 1000)

# 進行訓練
model_xgb = xgb_model.fit(X_train, y_train, verbose=True,
                          eval_set=[(X_train, y_train), (X_test, y_test)])

# 抓出重要特徵值的重要性
feat_imp = model_xgb.feature_importances_
feat_imp.sum()

# 抓出特徵值欄位
feat = X_train.columns.tolist()

# 合併變成DataFrame
res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp})

# 根據Importance，由高到低排序
res_data = res_data.sort_values(by='Importance', ascending=False)

# 視覺化

plot_name = 'xgb_Importances'
ax = sns.barplot(res_data['Importance'], res_data['Features'])
ax.set(xlabel=plot_name+' Feature Importance Score', ylabel=plot_name+' Features', title=plot_name)
ax.get_figure().savefig(plot_name+'.png', dpi=300)
res_data.to_csv(plot_name+'.csv' ,encoding = 'cp950')


# LGB
# 命名模型物件
lgb_model = lgb.LGBMClassifier(n_estimators = 1000,verbose=1)

# 進行訓練
lgb_model.fit(X_train, y_train)

# 抓出重要特徵值的重要性
feat_imp = lgb_model.feature_importances_
feat_imp.sum()

# 抓出特徵值欄位
feat = X_train.columns.tolist()

# 合併變成DataFrame
res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp})

# 根據Importance，由高到低排序
res_data = res_data.sort_values(by='Importance', ascending=False)

# 視覺化

plot_name = 'lgb_Importances'
ax = sns.barplot(res_data['Importance'], res_data['Features'])
ax.set(xlabel=plot_name+' Feature Importance Score', ylabel=plot_name+' Features', title=plot_name)
ax.get_figure().savefig(plot_name+'.png', dpi=300)
res_data.to_csv(plot_name+'.csv' ,encoding = 'cp950')

# RandomForest
# 命名模型物件
rf_model = RandomForestClassifier(n_estimators = 1000)

# 進行訓練
rf_model = rf_model.fit(X_train, y_train)

# 抓出重要特徵值的重要性
feat_imp = rf_model.feature_importances_
feat_imp.sum()

# 抓出特徵值欄位
feat = X_train.columns.tolist()

# 合併變成DataFrame
res_data = pd.DataFrame({'Features': feat, 
                         'Importance': feat_imp})

# 根據Importance，由高到低排序
res_data = res_data.sort_values(by='Importance', ascending=False)

# 視覺化

plot_name = 'rf_Importances'
ax = sns.barplot(res_data['Importance'], res_data['Features'])
ax.set(xlabel=plot_name+' Feature Importance Score', ylabel=plot_name+' Features', title=plot_name)
ax.get_figure().savefig(plot_name+'.png', dpi=300)
res_data.to_csv(plot_name+'.csv' ,encoding = 'cp950')

# LogisticRegression
# 命名模型物件
from sklearn.linear_model import LogisticRegression
logistic_reg =LogisticRegression()


# 進行訓練
logistic_reg.fit(X_train, y_train)
logistic_reg.coef_
X_train.columns


# 抓出重要特徵值的重要性
feat_imp = logistic_reg.coef_.tolist()[0]

# 抓出特徵值欄位
feat = X_train.columns.tolist()

# 合併變成DataFrame
res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp})

# 特徵值的正/負屬性高到低
res_data = res_data.sort_values(by='Importance', ascending=False)

# 知道單位增/減量
res_data['delta'] = np.exp(res_data['Importance'])-1

# 視覺化
plot_name = 'lr_Importances'
ax = sns.barplot(res_data['Importance'], res_data['Features'])
ax.set(xlabel=plot_name+' Feature Importance Score', ylabel=plot_name+' Features', title=plot_name)
ax.get_figure().savefig(plot_name+'.png', dpi=300)
res_data.to_csv(plot_name+'.csv' ,encoding = 'cp950')


#svm_model = SVC(kernel='linear')
#svm_model.fit(X_train, y_train)


# 評估
#XGB
print('XGB準確率: ',model_xgb.score(X_test,y_test))

#RF
print('RF訓練集: ',rf_model.score(X_test,y_test))

#LGB
print('LGB訓練集: ',lgb_model.score(X_test,y_test))


X_train.columns
