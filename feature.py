
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':int,'end_time':int})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':int})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[ ]:


voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':int,'end_time':int})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':int})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[ ]:


uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)


# In[ ]:


voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)


# ##  baseline 

# ### 通话记录

# In[ ]:

voice['call_time'] = voice['end_time'] - voice['start_time']


voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice_call_time=voice.groupby(['uid'])['call_time'].agg(['std','max','min','mean','sum','var','mad','median']).add_prefix('voice_call_time_').reset_index().fillna(0)


# ## 短信记录

# In[ ]:


sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)


sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


# ### 网站/APP记录

# In[ ]:


wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum','var','mad']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum','var','mad']).add_prefix('wa_visit_dura_').reset_index()


up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum','var','mad']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum','var','mad']).add_prefix('wa_down_flow_').reset_index()

wa_type = wa.groupby(['uid','wa_type'])['uid'].count().unstack().add_prefix('wa_type_').reset_index().fillna(0)

# In[ ]:


feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_call_time,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,wa_name,visit_cnt,visit_dura,up_flow,
           down_flow,wa_type]

# In[ ]:


train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')


# In[3]:


test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')


# In[4]:


train_feature.to_csv('../data/train_featureV1.csv',index=None)
test_feature.to_csv('../data/test_featureV1.csv',index=None)

