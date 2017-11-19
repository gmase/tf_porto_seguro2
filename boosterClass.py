# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:45:04 2017

@author: gmase
"""

import xgboost as xgb
import random
from datetime import datetime
import pandas as pd
import tensorflow as tf

rutaOrigen='//home//gmase//Documents//tensorflow//tf_porto_seguro2//origin//'
rutaProcesados='//home//gmase//Documents//tensorflow//tf_porto_seguro2//processed//'
rutaOutputXgboost='//home//gmase//Documents//tensorflow//tf_porto_seguro2//xgboost_output//'
rutaOutputDeep='//home//gmase//Documents//tensorflow//tf_porto_seguro2//deep_output//'

test_file_name=rutaOrigen+'test.csv'
train_file_name=rutaOrigen+'train.csv'

CSV_COLUMNS = [
    "id","reporta","ps_ind_01","ps_ind_02_cat","ps_ind_03","ps_ind_04_cat","ps_ind_05_cat","ps_ind_06_bin","ps_ind_07_bin","ps_ind_08_bin","ps_ind_09_bin","ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin","ps_ind_13_bin","ps_ind_14","ps_ind_15","ps_ind_16_bin","ps_ind_17_bin","ps_ind_18_bin","ps_reg_01","ps_reg_02","ps_reg_03","ps_car_01_cat","ps_car_02_cat","ps_car_03_cat","ps_car_04_cat","ps_car_05_cat","ps_car_06_cat","ps_car_07_cat","ps_car_08_cat","ps_car_09_cat","ps_car_10_cat","ps_car_11_cat","ps_car_11","ps_car_12","ps_car_13","ps_car_14","ps_car_15","ps_calc_01","ps_calc_02","ps_calc_03","ps_calc_04","ps_calc_05","ps_calc_06","ps_calc_07","ps_calc_08","ps_calc_09","ps_calc_10","ps_calc_11","ps_calc_12","ps_calc_13","ps_calc_14","ps_calc_15_bin","ps_calc_16_bin","ps_calc_17_bin","ps_calc_18_bin","ps_calc_19_bin","ps_calc_20_bin"
]

def train_test(dtrain,dtest,param,num_round):
    progress = dict()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    #,early_stopping_rounds=3
    bst = xgb.train(param, dtrain, num_round,evallist,evals_result=progress)
    #print('ultimo: {}'.format(bst.eval))
    return(progress['eval']['auc'][-1])
    

  
class Booster:
    def __init__(self, max_depth,eta,num_round,name):
        self.max_depth=max_depth
        self.eta=eta
        self.num_round=num_round
        self.param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': 1}
        self.param['nthread'] = 4
        self.param['eval_metric'] = 'auc'
        self.tested=1
        self.cummulative=0
        
        self.param['objective'] ='binary:logistic'        
        #self.param['objective'] = 'rank:pairwise'
        
        """
        self.param['objective'] = 'binary:logitraw'
        self.param['objective'] = 'multi:softmax'
        self.param['num_class'] = 2
        self.param['objective'] = 'multi:softprob'
        self.param['objective'] = 'reg:logistic'
        self.param['objective'] = 'count:poisson' 0.62932
        self.param['objective'] = 'rank:pairwise' 0.632882
        """

        self.value=0
        self.name="{}".format(name)
    def train_test(self,dtrain,dtest):
        self.tested+=1
        self.cummulative+=train_test(dtrain,dtest,self.param,self.num_round)
        self.value=1.0*self.cummulative/self.tested
        
        print(self.value)
        
    def updateName(self,add):
        self.name="{}_{}".format(self.name,add)
        
    def muestrate(self):
        print(' id: {} depth: {}  rounds: {}  eta: {}'.format(self.name,self.max_depth,self.num_round,self.eta))
        return ' id: {} depth: {}  rounds: {}  eta: {} auc: {}'.format(self.name,self.max_depth,self.num_round,self.eta,self.value)
        
    def mutate(self,mutation_factor):
        addM=False
        if random.random()<mutation_factor:
            addM=True
            if (int(round(random.random()*10)))%2==0:
                modifier=1
            else:
                modifier=-1
            self.max_depth=self.max_depth+modifier
        
        if random.random()<mutation_factor:
            addM=True
            if (int(round(random.random()*10)))%2==0:
                modifier=1
            else:
                modifier=-1
            self.num_round=int(round(self.num_round+(modifier*mutation_factor*10)))
        
        if random.random()<mutation_factor:
            addM=True
            if (int(round(random.random()*10)))%2==0:
                modifier=1
            else:
                modifier=-1
            self.eta=self.eta+(modifier*mutation_factor*0.1)
        
        if self.max_depth<=1:
            self.max_depth=2
        if self.eta<=0.01:
            self.eta=0.02
        if self.num_round<=2:
            self.num_round=3
            
        self.param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': 1}
        self.param['nthread'] = 4
        self.param['eval_metric'] = 'auc'
        self.param['objective'] = 'binary:logistic'

        
        if addM:
            self.updateName('M')


    def evaluate(self,dtrain_full,dtest_full,output_file):
        df_test = pd.read_csv(
          tf.gfile.Open(test_file_name),
          names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
        
        bst = xgb.train(self.param, dtrain_full, self.num_round)
        prediction=bst.predict(dtest_full)
        bst.save_model('0001.model')
        xgb.to_graphviz(bst, num_trees=2)   
        xgb.plot_importance(bst)
        xgb.plot_tree(bst, num_trees=2)
        
        timeStamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        f = open(output_file, 'w')
        f.write("id,target\n")
        i=0
        for i,p in enumerate(prediction):
              f.write("{},{}\n".format(df_test['id'][i],p))
        f.close()
        
    def evaluateTrain(self,dtrain_full,dtest_full,output_file):
        
        df_test = pd.read_csv(
          tf.gfile.Open(train_file_name),
          names=CSV_COLUMNS,
          skipinitialspace=True,
          engine="python",
          skiprows=1)
        
        bst = xgb.train(self.param, dtrain_full, self.num_round)
        prediction=bst.predict(dtest_full)
        bst.save_model('0001.model')
        
        
        timeStamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        f = open(output_file, 'w')
        f.write("id,target\n")
        i=0
        for i,p in enumerate(prediction):
              f.write("{},{}\n".format(df_test['id'][i],p))
              
        f.close()