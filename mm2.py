# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:50:35 2017

@author: gmase
"""

import pandas as pd
import sys

rutaOrigen='//home//gmase//Documents//tensorflow//tf_porto_seguro2//origin//'
rutaProcesados='//home//gmase//Documents//tensorflow//tf_porto_seguro2//processed//'
rutaOutputXgboost='//home//gmase//Documents//tensorflow//tf_porto_seguro2//xgboost_output//'
rutaOutputDeep='//home//gmase//Documents//tensorflow//tf_porto_seguro2//deep_output//'

test_file_name=rutaOrigen+'test.csv'
train_file_name=rutaOrigen+'train.csv'


df_xgboost = pd.read_csv(filepath_or_buffer=rutaOutputXgboost+'xgboost_para_mm2.csv')
df_deep = pd.read_csv(filepath_or_buffer=rutaOutputDeep+'deep_para_mm2.csv')
df_train = pd.read_csv(filepath_or_buffer=train_file_name)

print(df_train.describe())
print(df_xgboost.describe())
print(df_deep.describe())
sys.exit("Quieto parao")
factor1=7
factor2=4
#filtro=(df_xgboost['target']>(0.036469-0.019020/factor)) & (df_xgboost['target']<(0.036469+0.019020/factor))
filtro=((df_xgboost['target']>(0.036469-0.019020/factor1)) & (df_xgboost['target']<(0.036469+0.019020/factor1)) 
    & ((df_deep['target']<(0.027865-0.015294/factor2)) | (df_deep['target']>(0.027865+0.015294/factor2))))

df_xgboost_doubts=df_xgboost[filtro]
df_deep_doubts=df_deep[filtro]
print(df_xgboost_doubts.describe())
print(df_deep_doubts.describe())


df_xgboost_deep=df_xgboost.copy(deep=True)
df_xgboost_deep.loc[filtro,'target']=df_deep[filtro]['target']+0.036469-0.027865
df_xgboost_deep.to_csv(path_or_buf=rutaOutputXgboost+'mm2.csv',index=False,columns=['id','target'])
print(df_xgboost_deep.describe())