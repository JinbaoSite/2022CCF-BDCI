#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


# In[2]:


train_data = pd.read_csv('dataTrain.csv')
test_data = pd.read_csv('dataA.csv')
submission = pd.read_csv('submit_example_A.csv')
data_nolabel = pd.read_csv('dataNoLabel.csv')


# `train.csv`:包含全量数据集的70%（dataNoLabel是训练集的一部分，选手可以自己决定是否使用）
# 
# `test.csv`:包含全量数据集的30%
# 
# 位置类特特征：基于联通基站产生的用户信令数据；`f1~f6`
# 
# 互联网类特征：基于联通用户上网产生的上网行为数据； `f7~f42`
# 
# 通话类特征：基于联通用户日常通话、短信产生的数据`f43~f46`

# In[3]:


print(f'train_data.shape = {train_data.shape}\ntest_data.shape  = {test_data.shape}')


# In[4]:


train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']
# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)


# In[5]:


cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])


# In[6]:


num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]


# In[7]:


train = train[:50000]
label = label[:50000]


# In[8]:


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)

    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        print("Model = %s, KFold = %d, val_auc = %.4f" % (model_name, k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


# In[9]:


gbc = GradientBoostingClassifier(
    n_estimators=50, 
    learning_rate=0.1,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6, 
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05, 
    n_estimators=100, 
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210, 
    depth=6, 
    learning_rate=0.03, 
    l2_leaf_reg=1, 
    loss_function='Logloss', 
    verbose=0
)


# In[10]:


estimators = [
    ('gbc', gbc),
    ('hgbc', hgbc),
    ('xgbc', xgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression()
)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)


# In[12]:


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


# In[13]:


features = []
feature_importances = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        features.append(col)
    feature_importances.append([col, auc1, auc1 - auc])


# In[14]:


feature_importances.sort(key=lambda x: x[2])
for fi in feature_importances:
    print("| %10s | %.8f | %.8f |" % (fi[0], fi[1], fi[2]))


# In[15]:


clf.fit(X_train[features], y_train)
y_pred = clf.predict_proba(X_test[features])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


# In[16]:


train = train[features]
test = test[features]
preds = model_train(clf, "StackingClassifier", 10)


# In[17]:


submission['label'] = preds


# In[18]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




