
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('../data/train_featureV1.csv')
test = pd.read_csv('../data/test_featureV1.csv')


# In[3]:


dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# In[4]:


lgb_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': ('multi_logloss', 'multi_error'),
    #'metric_freq': 100,
    'is_training_metric': False,
    'min_data_in_leaf': 12,
    'num_leaves': 63,
    'learning_rate': 0.07,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity':1,
#    'gpu_device_id':2,
#    'device':'gpu'
#    'lambda_l1': 0.001,
#    'skip_drop': 0.95,
#    'max_drop' : 10
    #'lambda_l2': 0.005
    #'num_threads': 18
}    


# In[5]:


def evalMetric(preds,dtrain):
    
    label = dtrain.get_label()
    
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = 0.6*auc +0.4*f1
    
    return 'res',res,True
    

    


# ### 本地CV

# In[6]:


lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=10,verbose_eval=5,num_boost_round=1000,nfold=3,metrics=['evalMetric'])


# ## 训练

# In[7]:


model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain])


# ### 预测

# In[ ]:


pred=model.predict(test.drop(['uid'],axis=1))


# In[ ]:


res =pd.DataFrame({'uid':test.uid,'label':pred})


# In[ ]:


res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.5 else 0)
res.label = res.label.map(lambda x: int(x))


# In[ ]:


res.to_csv('../result/lgb-baseline.csv',index=False,header=False,sep=',',columns=['uid','label'])

