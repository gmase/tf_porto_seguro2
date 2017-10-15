# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import itertools

import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow as tf
from datetime import datetime
import xgboost as xgb
import random
import copy

rutaOrigen='//home//gmase//Documents//tensorflow//tf_porto_seguro2//origin//'
rutaProcesados='//home//gmase//Documents//tensorflow//tf_porto_seguro2//processed//'
rutaOutputXgboost='//home//gmase//Documents//tensorflow//tf_porto_seguro2//xgboost_output//'
rutaOutputDeep='//home//gmase//Documents//tensorflow//tf_porto_seguro2//deep_output//'

test_file_name=rutaOrigen+'test.csv'
train_file_name=rutaOrigen+'train.csv'



CSV_COLUMNS = [
    "id","reporta","ps_ind_01","ps_ind_02_cat","ps_ind_03","ps_ind_04_cat","ps_ind_05_cat","ps_ind_06_bin","ps_ind_07_bin","ps_ind_08_bin","ps_ind_09_bin","ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin","ps_ind_13_bin","ps_ind_14","ps_ind_15","ps_ind_16_bin","ps_ind_17_bin","ps_ind_18_bin","ps_reg_01","ps_reg_02","ps_reg_03","ps_car_01_cat","ps_car_02_cat","ps_car_03_cat","ps_car_04_cat","ps_car_05_cat","ps_car_06_cat","ps_car_07_cat","ps_car_08_cat","ps_car_09_cat","ps_car_10_cat","ps_car_11_cat","ps_car_11","ps_car_12","ps_car_13","ps_car_14","ps_car_15","ps_calc_01","ps_calc_02","ps_calc_03","ps_calc_04","ps_calc_05","ps_calc_06","ps_calc_07","ps_calc_08","ps_calc_09","ps_calc_10","ps_calc_11","ps_calc_12","ps_calc_13","ps_calc_14","ps_calc_15_bin","ps_calc_16_bin","ps_calc_17_bin","ps_calc_18_bin","ps_calc_19_bin","ps_calc_20_bin"
]


CATEGORICAL_COLS=[
"ps_ind_02_cat",
"ps_ind_04_cat",
"ps_ind_05_cat",
"ps_car_01_cat",
"ps_car_02_cat",
"ps_car_03_cat",
"ps_car_04_cat",
"ps_car_05_cat",
"ps_car_06_cat",
"ps_car_07_cat",
"ps_car_08_cat",
"ps_car_09_cat",
"ps_car_10_cat",
"ps_car_11_cat"]

"""
CATEGORICAL_COLS=[
"ps_ind_02_cat",
"ps_ind_04_cat",
"ps_ind_05_cat",
"ps_car_01_cat",
"ps_car_02_cat",
"ps_car_03_cat",
"ps_car_04_cat",
"ps_car_05_cat",
"ps_car_06_cat",
"ps_car_07_cat",
"ps_car_08_cat",
"ps_car_09_cat",
"ps_car_10_cat"]
"""



FLAGS = None

def createBinTest():
    df_train = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
      skipinitialspace=True,
      engine="python",
      skiprows=1)
      
    df_train = pd.get_dummies( df_train,columns = CATEGORICAL_COLS ).as_matrix()
    data=df_train[:,1:]
    dtrain = xgb.DMatrix(data, label=None)
    dtrain.save_binary(rutaProcesados+'test_full.buffer')

        
def createBin(div,name):
    

    df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
      
    df_train = pd.get_dummies( df_train,columns = CATEGORICAL_COLS).as_matrix()
        
    if name=='_0':
        np.random.seed(42)
    msk = np.random.rand(len(df_train)) < div
    train = df_train[msk]
    test = df_train[~msk]
    
    if div==1:
        data=train[:,2:]
        label=train[:,1]
        dtrain = xgb.DMatrix(data, label=label)
        dtrain.save_binary(rutaProcesados+'train_full.buffer')
        
    else:
        data=train[:,2:]
        label=train[:,1]
    
        data_test=test[:,2:]
        label_test=test[:,1]

        dtrain = xgb.DMatrix(data, label=label)
        dtest= xgb.DMatrix(data_test, label=label_test)
        
        dtrain.save_binary(rutaProcesados+'train_train'+name+'.buffer')
        dtest.save_binary(rutaProcesados+'train_test'+name+'.buffer')


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
        self.param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': 1, 'objective': 'binary:logistic'}
        self.param['nthread'] = 4
        self.param['eval_metric'] = 'auc'
        self.value=-1
        self.name="{}".format(name)
    def train_test(self,dtrain,dtest):
        self.value=train_test(dtrain,dtest,self.param,self.num_round)
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
            
        self.param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': 1, 'objective': 'binary:logistic'}
        self.param['nthread'] = 4
        self.param['eval_metric'] = 'auc'
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

    
def getOutput(depth,rounds,eta,train,test,name):
    adan=Booster(depth,eta,rounds,'no_name')
    adan.evaluate(train,test,name)
    
    
def createBuffers():
    createBin(0.8,'_0')
    createBin(0.8,'_1')
    createBin(0.8,'_2')
    createBin(1,'na')
    
    createBinTest()
    
def evolve():
    dtrain = [xgb.DMatrix(rutaProcesados+'train_train_0.buffer'),xgb.DMatrix(rutaProcesados+'train_train_1.buffer'),xgb.DMatrix(rutaProcesados+'train_train_2.buffer')]
    dtest = [xgb.DMatrix(rutaProcesados+'train_test_0.buffer'),xgb.DMatrix(rutaProcesados+'train_test_1.buffer'),xgb.DMatrix(rutaProcesados+'train_test_2.buffer')]

    mutation_factor=0.3
    population=[Booster(4,0.3,20,i) for i in range(10)]
    population.extend([Booster(12,0.1,30,i+10) for i in range(5)])
    population.extend([Booster(6,0.28,15,i+4) for i in range(4)])
    population.append(Booster(5,0.3,30,19))
    for i in population:
        i.mutate(mutation_factor)
    
    f = open('eval_log.csv', 'a')
    f.write('\n')
 
    generation_number=10 
    
    for gen in range(generation_number):
        for i in population:
            i.train_test(dtrain[gen%3],dtest[gen%3])
        population.sort(key=lambda x: x.value, reverse=True)    
        print("GENERACION {}\n\n".format(gen))
        f.write("\n\nGENERACION {}\n".format(gen))
        for n,item in enumerate(population):
            print(item.value)
            item.updateName(n)
            f.write('{} -- {}'.format(item.value,item.muestrate()))
            f.write('\n')
        
        for i in range(3):
            for j,x in enumerate(population[:4]):
                population[-1-(j*i)]=copy.deepcopy(x)
                   
        for i in population[3:]:
            i.mutate(mutation_factor)
        
    population[0].muestrate()
    f.close()
    
def main(_):

    #Step 1 create buffer files

    #createBuffers()


    
    #Step 2 evolve
    evolve()
    
    
    sys.exit("Quieto parao")

    #Step 3 do predict

    #adan=Booster(3,0.3,28)
    #adan.train_test(dtrain[0],dtest[0])
    #depth: 6  rounds: 15  eta: 0.28
    #depth: 6  rounds: 26  eta: 0.26
    #depth: 5  rounds: 38  eta: 0.22
    #depth: 2  rounds: 20  eta: 0.3 auc: 0.632719 caca
    #depth: 4  rounds: 39  eta: 0.3 auc: 0.64922

    dtrain_full = xgb.DMatrix(rutaProcesados+'train_full.buffer')
    dtest_full = xgb.DMatrix(rutaProcesados+'test_full.buffer')
    getOutput(5,38,0.22,dtrain_full,dtest_full,rutaOutputXgboost+'xgboost_train_full.csv')
    

    #w = np.random.rand(5, 1)
    #dtrain = xgb.DMatrix(data, label=label, missing=-999.0, weight=w)


    

    
                 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="deep2",
      help="Valid model types: {'wide', 'deep','deep2', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--just_test",
      type=str,
      default="yes",
      help="Valid model types: {'yes', 'no'}."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  parser.add_argument(
      "--learning_rate",
      type=str,
      default=0.06,
      help="Learning rate."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)