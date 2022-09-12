#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_data_path = 'data/train.csv'
test_data_path = 'data/evaluation_public.csv'
submission_path = 'data/submit_example.csv'


# In[3]:


train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
print(f'train_data.shape = {train_data.shape}, test_data.shape = {test_data.shape}')


# In[4]:


train_data['is_risk'].value_counts()


# In[5]:


train_data.head()


# In[6]:


train_data.groupby(['user_name']).agg({
    'device_num_transform': pd.Series.nunique, 
    'ip_transform': pd.Series.nunique,
    'browser_version': pd.Series.nunique, 
    'browser': pd.Series.nunique,
    'os_type': pd.Series.nunique, 
    'os_version': pd.Series.nunique,
    'ip_type': pd.Series.nunique,
    'http_status_code': pd.Series.nunique, 
    'op_city': pd.Series.nunique,
    'log_system_transform': pd.Series.nunique, 
    'url': pd.Series.nunique,
})


# In[7]:


# from: https://zhuanlan.zhihu.com/p/463778333
test_data['is_risk'] = -1
data = pd.concat([train_data, test_data])
data['op_datetime'] = pd.to_datetime(data['op_datetime'])
# day, hour, minute
data['timestamp'] = data["op_datetime"].values.astype(np.int64) // 10 ** 9
data['day'] = data['op_datetime'].dt.day
data['hour'] = data['op_datetime'].dt.hour
data['minute'] = data['op_datetime'].dt.minute

data['day_sin'] = np.sin(2 * np.pi * data['day']/24.0) 
data['day_cos'] = np.cos(2 * np.pi * data['day']/24.0)
data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24.0) 
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24.0)
data['minute_sin'] = np.sin(2 * np.pi * data['minute']/60.0) 
data['minute_cos'] = np.cos(2 * np.pi * data['minute']/60.0)


# In[8]:


data.head()


# In[9]:


cat_columns = [
    'user_name', 'department', 'ip_transform', 'device_num_transform',
    'browser_version', 'browser', 'os_type', 'os_version',
    'ip_type', 'http_status_code', 'op_city', 'log_system_transform', 'url'
]


# In[10]:


record = dict()
res = dict()
data = data.sort_values(by=['user_name', 'timestamp']).reset_index(drop=True)
for idx, row in tqdm(data.iterrows()):
    user_name = row['user_name']
    for col in cat_columns:
        key = str(user_name) + "_" + str(row[col])
        
        if key not in record:
            record[key] = [row['timestamp']]
        else:
            record[key].append(row['timestamp'])
        for idx in range(1, 4):
            column = f'user_name_{col}_diff_{idx}'
            if column not in res:
                res[column] = [0]
            else:
                if len(record[key]) < idx + 1:
                    res[column].append(0)
                else:
                    res[column].append(row['timestamp'] - record[key][-(idx+1)])
for key in res.keys():
    data[key] = res[key]


# In[11]:


num_columns = [col for col in data.columns if col not in cat_columns and 
                col not in ['id', 'op_datetime', 'op_month', 'timestamp', 'is_risk']]
target = 'is_risk'
feature = cat_columns + num_columns


# In[12]:


for col in cat_columns:
    lab = LabelEncoder()
    data[col] = lab.fit_transform(data[col])


# In[13]:


x_train = data[(data['is_risk'] != -1) & (data['op_month'] != '2022-04')][feature]
y_train = data[(data['is_risk'] != -1) & (data['op_month'] != '2022-04')][target]
x_val = data[(data['is_risk'] != -1) & (data['op_month'] == '2022-04')][feature]
y_val = data[(data['is_risk'] != -1) & (data['op_month'] == '2022-04')][target]


# In[14]:


x_test = data[data['is_risk'] == -1][feature]
train = data[data['is_risk'] != -1][feature]
label = data[data['is_risk'] != -1][target]


# In[15]:


data['is_risk'].value_counts()


# In[16]:


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train_data.shape[0]))
    test_preds = np.zeros(test_data.shape[0])
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    print(f'Model = {model_name}')
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_val = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_val = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train, y_train)

        y_pred = model.predict_proba(x_val)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_val, y_pred)
        print("KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(x_test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


# In[17]:


xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=25, 
    max_depth=6, 
    learning_rate=0.1
)
xgbc_test_preds = model_train(xgbc, "XGBClassifier", 10)


# In[18]:


gbm = LGBMClassifier(
    objective='binary',
    num_leaves=35, 
    learning_rate=0.1, 
    n_estimators=100, 
    metrics='auc'
)
gbm_test_preds = model_train(gbm, "LGBMClassifier", 10)


# In[19]:


cbc = CatBoostClassifier(
    iterations=100, 
    depth=10, 
    learning_rate=0.1, 
    loss_function='Logloss',
    verbose=0
)
cbc_test_preds = model_train(cbc, "CatBoostClassifier", 10)


# In[20]:


preds = (xgbc_test_preds + gbm_test_preds + cbc_test_preds) / 3


# In[21]:


submission = pd.DataFrame({
    'id': data[data['is_risk'] == -1]['id'],
    'is_risk': np.array(preds)
}).sort_values(by=['id']).reset_index(drop=True)


# In[22]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




