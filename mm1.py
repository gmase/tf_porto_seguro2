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
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import copy


rutaOrigen='//home//gmase//Documents//tensorflow//tf_porto_seguro2//origin//'
rutaProcesados='//home//gmase//Documents//tensorflow//tf_porto_seguro2//processed//'
rutaOutputXgboost='//home//gmase//Documents//tensorflow//tf_porto_seguro2//xgboost_output//'
rutaOutputDeep='//home//gmase//Documents//tensorflow//tf_porto_seguro2//deep_output//'
rutaOutputMeta='//home//gmase//Documents//tensorflow//tf_porto_seguro2//meta_output//'

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


        
def createBin(div,dtrain_full):
      
    dtrain = xgb.DMatrix('//home//gmase//Documents//tensorflow//tf_porto_seguro//xgboost//train_0.buffer')
    dtest = xgb.DMatrix('//home//gmase//Documents//tensorflow//tf_porto_seguro//xgboost//test_0.buffer')
    
    
    msk = np.random.rand(len(df_train)) < div
    train = dtrain_full[msk]
    test = dtrain_full[~msk]
    
    data=train[:,2:]
    label=train[:,1]
    
    data_test=test[:,2:]
    label_test=test[:,1]
        
    dtrain = xgb.DMatrix(data, label=label)
    dtest=xgb.DMatrix(data_test, label=label_test)
        
    dtrain.save_binary('Mtrain.buffer')
    dtest.save_binary('Mtest.buffer')


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
        #self.muestrate()
    def evaluate(self,dtrain_full,dtest_full,output_file):

        
        bst = xgb.train(self.param, dtrain_full, self.num_round)
        prediction=bst.predict(dtest_full)
        bst.save_model('0001.model')
        
        
        test_file_name=rutaOrigen+'test.csv'
        df_test = pd.read_csv(
          tf.gfile.Open(test_file_name),
          names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
        f = open(output_file, 'w')
        f.write("id,target\n")
        i=0
        for i,p in enumerate(prediction):
              f.write("{},{}\n".format(df_test['id'][i],p))
        f.close()
        

    
    #file1=xgboost
def compare_results(file1,file2,truth):
   df_1 = pd.read_csv(
          tf.gfile.Open(file1),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
   df_2 = pd.read_csv(
          tf.gfile.Open(file2),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
   df_truth = pd.read_csv(
          tf.gfile.Open(truth),
          names=CSV_COLUMNS,
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
   df_compare=(df_truth['reporta']-df_1['pred'])**2-((df_truth['reporta']-df_2['pred'])**2)

   #0 usa deep, 1 usa xgboost
   difer=df_compare.iloc[:]>-0.05
   print(difer.shape)
   print(difer.count())
   
   dtrain_full = xgb.DMatrix('//home//gmase//Documents//tensorflow//tf_porto_seguro//xgboost//train_full.buffer')
   dtrain_full.set_label(difer)
   
   dtrain_full.save_binary('meta_train_full.buffer')

    #Cambio de label para conjunto test 0.8
   np.random.seed(42)
   msk = np.random.rand(len(difer)) < 0.8
   label_train = difer[msk]
   label_test = difer[~msk]
    
   dtrain = xgb.DMatrix('//home//gmase//Documents//tensorflow//tf_porto_seguro//xgboost//train_0.buffer')
   dtrain.set_label(label_train)
   dtrain.save_binary('meta_part_train.buffer')
   dtest = xgb.DMatrix('//home//gmase//Documents//tensorflow//tf_porto_seguro//xgboost//test_0.buffer')
   dtest.set_label(label_test)
   dtest.save_binary('meta_part_test.buffer')
   
   
def subConjuntoParaDeep():
    dtrain_full = xgb.DMatrix(rutaOutputMeta+'meta_train_full.buffer')
    dtest = xgb.DMatrix(rutaProcesados+'train_full.buffer',)
    
    adan=Booster(13,0.13,30,'no_name')
    adan.evaluate(dtrain_full,dtest,rutaOutputMeta+"mm_para_deep.csv")
    df_1 = pd.read_csv(
          tf.gfile.Open(rutaOutputMeta+"mm_para_deep.csv"),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
        
    mask_for_deep=df_1<0.8
    train_for_deep = df_train[mask_for_deep]
    train_for_deep.to_csv(path_or_buf=rutaOutputMeta+'meta_for_deep.csv',sep=',')

    
    
def componeSolucionFinal(deep_name,xgboost_name,choose_name):
   df_1 = pd.read_csv(
          tf.gfile.Open(xgboost_name),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
   df_2 = pd.read_csv(
          tf.gfile.Open(deep_name),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
   df_choose = pd.read_csv(
          tf.gfile.Open(choose_name),
          names=['id','choose'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
   
   #TODO normalizar?
   
   cuenta_deep=0
   cuenta_boost=0
   f = open(rutaOutputMeta+'META_output.csv', 'w')
   f.write("id,target\n")
   for v1,v2,c in zip(df_1.values,df_2.values,df_choose['choose']):
       f.write("{},".format(int(v1[0])))
       if c>=0.5:
           val=(v1[1]*3.0+v2[1]*2.0)/5.0
           cuenta_boost+=1
       else:
           val=(v1[1]*2.0+3.0*v2[1])/5.0
           cuenta_deep+=1
       f.write("{}\n".format(val))
   f.close()
   print('De boost salen: {}  De deep: {}'.format(cuenta_boost,cuenta_deep))
          
   """      
   f = open(rutaOutputMeta+'META_output.csv', 'w')
   f.write("id,target\n")
   for v1,v2,c in zip(df_1.values,df_2.values,df_choose['choose']):
       f.write("{},".format(int(v1[0])))
       if c>=0.8:
           f.write("{}\n".format(v1[1]))
           cuenta_boost+=1
       else:
           f.write("{}\n".format(v2[1]))
           cuenta_deep+=1
   """          
          
def genOutput():
    dtrain_full = xgb.DMatrix(rutaOutputMeta+'meta_train_full.buffer')
    dtest_full = xgb.DMatrix(rutaProcesados+'test_full.buffer',)
    
    adan=Booster(13,0.13,30,'no_name')
    adan.evaluate(dtrain_full,dtest_full,rutaOutputMeta+"mm_final.csv")
    componeSolucionFinal(rutaOutputDeep+'deep_full.csv',rutaOutputXgboost+'xgboost_full.csv',rutaOutputMeta+"mm_final.csv")
    
    
def trainMetaModel(generaciones):
    dtrain = xgb.DMatrix(rutaOutputMeta+'meta_part_train.buffer')
    dtest = xgb.DMatrix(rutaOutputMeta+'meta_part_test.buffer')
    
    #depth: 6  rounds: 30  eta: 0.33 auc: 0.932554
    
    population=[Booster(6,0.33,30,i) for i in range(10)]
    population.extend([Booster(12,0.1,30,i+10) for i in range(5)])
    population.extend([Booster(6,0.28,15,i+4) for i in range(4)])
    population.append(Booster(5,0.3,30,19))
    
    mutation_factor=0.3
    for i in population:
        i.mutate(mutation_factor)
    
    
    f = open('eval_log.csv', 'a')
    f.write('\n')
 
    generation_number=generaciones
    
    for gen in range(generation_number):
        for i in population:
            i.train_test(dtrain,dtest)
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
    
    #file1=xgboost
def xgboostFails(file1,truth):
   df_1 = pd.read_csv(
          tf.gfile.Open(file1),
          names=['id','pred'],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
   df_truth = pd.read_csv(
          tf.gfile.Open(truth),
          names=CSV_COLUMNS,
          skipinitialspace=True,
          engine="python",
          skiprows=1)
          
   df_compare=(df_truth['reporta']-df_1['pred'])**2
   difer=df_compare.iloc[:]<0.3
   print (difer)
   
   dtrain_full = xgb.DMatrix(rutaProcesados+'train_full.buffer')
   dtrain_full.set_label(difer)
   
   dtrain_full.save_binary(rutaOutputMeta+'meta_train_full.buffer')

    #Cambio de label para conjunto test 0.8
   np.random.seed(42)
   msk = np.random.rand(len(difer)) < 0.8
   label_train = difer[msk]
   label_test = difer[~msk]
    
   dtrain = xgb.DMatrix(rutaProcesados+'train_train_0.buffer')
   dtrain.set_label(label_train)
   dtrain.save_binary(rutaOutputMeta+'meta_part_train.buffer')
   dtest = xgb.DMatrix(rutaProcesados+'train_test_0.buffer')
   dtest.set_label(label_test)
   dtest.save_binary(rutaOutputMeta+'meta_part_test.buffer')
   
def main(_):
    #Step1 prepare the files
    #xgboostFails(rutaOutputXgboost+'xgboost_train_full.csv',train_file_name)
    
    #Step2 train the model
    #trainMetaModel(1)
    
    #Step2 compose the output
    #genOutput()
    
    #compare_results(rutaOutputXgboost+'xgboost_train_full.csv',rutaOutputDeep+'deep_train_full.csv',train_file_name)

    #trainMetaModel(5)
    genOutput()

    #subConjuntoParaDeep()

    


    
                 

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
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
  
  