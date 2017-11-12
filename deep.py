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

import pVariable
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

data = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)


# Transformations.
#age_buckets = tf.feature_column.bucketized_column(
#    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Wide columns and deep columns.
"""
base_columns = [
   ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin,ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin,ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01,ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat,ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11,ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04,ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12,ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin,ps_calc_19_bin,ps_calc_20_bin,
]
"""

"""
    tf.feature_column.embedding_column(ps_ind_02_cat, dimension=8),
    tf.feature_column.embedding_column(ps_ind_04_cat, dimension=8),
    tf.feature_column.embedding_column(ps_ind_05_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_01_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_02_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_03_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_04_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_05_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_06_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_07_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_08_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_09_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_10_cat, dimension=8),
    tf.feature_column.embedding_column(ps_car_11_cat, dimension=8),
    
    #tf.feature_column.indicator_column(workclass),
    # To show an example of embedding
    tf.feature_column.embedding_column(ps_ind_02_cat, dimensionForEmbedding[0]),
    tf.feature_column.embedding_column(ps_ind_04_cat, dimensionForEmbedding[1]),
    tf.feature_column.embedding_column(ps_ind_05_cat, dimensionForEmbedding[2]),
    tf.feature_column.embedding_column(ps_car_01_cat, dimensionForEmbedding[3]),
    tf.feature_column.embedding_column(ps_car_02_cat, dimensionForEmbedding[4]),
    tf.feature_column.embedding_column(ps_car_03_cat, dimensionForEmbedding[5]),
    tf.feature_column.embedding_column(ps_car_04_cat, dimensionForEmbedding[6]),
    tf.feature_column.embedding_column(ps_car_05_cat, dimensionForEmbedding[7]),
    tf.feature_column.embedding_column(ps_car_06_cat, dimensionForEmbedding[8]),
    tf.feature_column.embedding_column(ps_car_07_cat, dimensionForEmbedding[9]),
    tf.feature_column.embedding_column(ps_car_08_cat, dimensionForEmbedding[10]),
    tf.feature_column.embedding_column(ps_car_09_cat, dimensionForEmbedding[11]),
    tf.feature_column.embedding_column(ps_car_10_cat, dimensionForEmbedding[12]),
    tf.feature_column.embedding_column(ps_car_11_cat, dimensionForEmbedding[13]),
    """

dimensionForEmbedding=[3,2,4,5,2,4,2,5,2,2,2,3,3,10]

"""
    tf.feature_column.indicator_column(ps_car_01_cat),
    tf.feature_column.indicator_column(ps_car_02_cat),
    tf.feature_column.indicator_column(ps_car_03_cat),
    tf.feature_column.indicator_column(ps_car_04_cat),
    #tf.feature_column.indicator_column(ps_car_05_cat),
    tf.feature_column.embedding_column(ps_car_05_cat, dimensionForEmbedding[7]),
    tf.feature_column.indicator_column(ps_car_06_cat),
    tf.feature_column.indicator_column(ps_car_07_cat),
    tf.feature_column.indicator_column(ps_car_08_cat),
    tf.feature_column.indicator_column(ps_car_09_cat),
    tf.feature_column.indicator_column(ps_car_10_cat),
    #tf.feature_column.indicator_column(ps_car_11_cat),
    tf.feature_column.embedding_column(ps_car_11_cat, dimensionForEmbedding[13]),
"""
    
        
def build_estimator(model_dir, model_type,learning_rate,layers,deep_columns):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns)
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
  #deep2 seems slightly better
  elif model_type=="deep2":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=layers,
        optimizer=tf.train.AdagradOptimizer( learning_rate=learning_rate)
    )
  return m


def input_fn(df_data, num_epochs, shuffle,num_threads):
  labels = df_data["reporta"].astype(bool)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=num_threads)
      
def input_predict(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=None,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=1)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data,just_test,learning_rate,layers,deep_columns):
  """Train and evaluate the model."""  
  #test_file_name='train.csv'
  train_df = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  train_df = train_df.dropna(how="any", axis=0)
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  m = build_estimator(model_dir, model_type,learning_rate,layers,deep_columns)

  if just_test=="yes":
        X_train, X_test = train_test_split(train_df, test_size=0.2, random_state=42)
        # set num_epochs to None to get infinite stream of data.
        m.train(
            input_fn=input_fn(X_train, num_epochs=None,num_threads=5, shuffle=True),
          steps=train_steps)
  # set steps to None to run evaluation until all data consumed.

        results = m.evaluate(
          input_fn=input_fn(X_test, num_epochs=1,num_threads=1, shuffle=False),
          steps=None)
        #print("model directory = %s" % model_dir)
        #for key in sorted(results):
        #  print("%s: %s" % (key, results[key]))

        timeStamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        f = open(rutaOutputDeep+'eval_log.csv', 'a')
        f.write(timeStamp)
        f.write('\n')
        f.write("LearningRate: {}".format(learning_rate))
        f.write("Layers: {}".format(layers))
        i=0
        for key in sorted(results):
          f.write("%s: %s\n" % (key, results[key]))
        f.close()
        return results['auc']
      
  else:
      m.train(
            input_fn=input_fn(train_df, num_epochs=None,num_threads=5, shuffle=True),
          steps=train_steps)
      y = m.predict(input_fn=input_predict(test_file_name, num_epochs=1, shuffle=False))
      
      
      df_test = pd.read_csv(
          tf.gfile.Open(test_file_name),
          names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
          skipinitialspace=True,
          engine="python",
          skiprows=1)
      """
      df_test = pd.read_csv(
          tf.gfile.Open(train_file_name),
          names=CSV_COLUMNS,
          skipinitialspace=True,
          engine="python",
          skiprows=1)
      """
    
      timeStamp=datetime.now().strftime('%Y%m%d_%H%M%S')
      f = open(rutaOutputDeep+'output'+timeStamp+'.csv', 'w')
      f.write("id,target\n")
      for i,p in enumerate(y):
          f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
          
      f.close()
      return 0

  
  
  #predictions = list(p["reporta"] for p in itertools.islice(y, 1))
  #print("Predictions: {}".format(str(predictions)))

def train_and_evalMM(model_dir, model_type, train_steps, train_data, test_data,just_test,learning_rate,layers):
    

    train_df = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    train_df = train_df.dropna(how="any", axis=0)
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir, model_type,learning_rate,layers)

    m.train(
          input_fn=input_fn(train_df, num_epochs=None,num_threads=5, shuffle=True),
        steps=train_steps)
        
        

    y = m.predict(input_fn=input_predict(test_file_name, num_epochs=1, shuffle=False))
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
        skipinitialspace=True,
        engine="python",
        skiprows=1)

    f = open(rutaOutputDeep+'deep_full.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
        
    f.close()
  
  
    y = m.predict(input_fn=input_predict(train_file_name, num_epochs=1, shuffle=False))

    df_test = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    
    f = open(rutaOutputDeep+'deep_train_full.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
        
    f.close()  
    
    return 0

def train_and_evalOnlyMM(model_dir, model_type, train_steps, train_data, test_data,just_test,learning_rate,layers):
    train_df = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    train_df = train_df.dropna(how="any", axis=0)
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir, model_type,learning_rate,layers)

    m.train(
          input_fn=input_fn(train_df, num_epochs=None,num_threads=5, shuffle=True),
        steps=train_steps)
        
        

    y = m.predict(input_fn=input_predict(test_file_name, num_epochs=1, shuffle=False))
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
        skipinitialspace=True,
        engine="python",
        skiprows=1)

    f = open(rutaOutputDeep+'deep_full.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
        
    f.close()
  
  
    y = m.predict(input_fn=input_predict(train_file_name, num_epochs=1, shuffle=False))

    df_test = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    
    f = open(rutaOutputDeep+'deep_train_full.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
        
    f.close()  
    
    return 0

def train_and_evalSubsetMM(model_dir, model_type, train_steps, train_data, test_data,just_test,learning_rate,layers):
    train_df = pd.read_csv(
        tf.gfile.Open(rutaOutputMeta+'meta_for_deep.csv'),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    train_df = train_df.dropna(how="any", axis=0)
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir, model_type,learning_rate,layers)

    m.train(
          input_fn=input_fn(train_df, num_epochs=None,num_threads=5, shuffle=True),
        steps=train_steps)

    y = m.predict(input_fn=input_predict(test_file_name, num_epochs=1, shuffle=False))
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=[]+CSV_COLUMNS[0:1]+CSV_COLUMNS[2:],
        skipinitialspace=True,
        engine="python",
        skiprows=1)

    f = open(rutaOutputDeep+'deep_from_mm.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(df_test['id'][i],p["probabilities"][1]))
        
    f.close()
  
    return 0
    
def train_and_evalSubsetMM2(model_dir, model_type, train_steps, train_data, test_data,just_test,learning_rate,layers,deep_columns):
    train_df_in = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    np.random.seed(42)
    msk = np.random.rand(len(train_df_in)) < 0.8
    train = train_df_in[msk]
    #test = train_df[~msk]
    # remove NaN elements
    train_df = train.dropna(how="any", axis=0)
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir, model_type,learning_rate,layers,deep_columns)

    m.train(
          input_fn=input_fn(train_df, num_epochs=None,num_threads=5, shuffle=True),
        steps=train_steps)

    y = m.predict(input_fn=input_predict(train_file_name, num_epochs=1, shuffle=False))

    f = open(rutaOutputDeep+'deep_para_mm2.csv', 'w')
    f.write("id,target\n")
    for i,p in enumerate(y):
        f.write("{},{}\n".format(train['id'][i],p["probabilities"][1]))
        
    f.close()
    

    
FLAGS = None


def main(_):
    f = open(rutaOutputDeep+'eval_log.csv', 'a')
    
    """
    train_and_evalOnlyMM(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256])    
    
    
    train_and_evalMM(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256])
    sys.exit("Quieto parao")
    
    """
    variables=pVariable.pVariables(CSV_COLUMNS[2:],data,True)
    deep_columns = [i.getTfVariable() for i in variables.vars]
    
    
    train_and_evalSubsetMM2(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256],deep_columns)
    sys.exit("Quieto parao")
    
    
    res=[]
    repeticiones=1
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256],deep_columns))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
    prev_auc=auc
    
    """
    for numIter in range(10):
        deep_columns = [i.getTfVariable() for i in variables.getSubset(0.9)]
        res=[]
        repeticiones=2
        for i in range(repeticiones):
              res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                     FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256],deep_columns))
        auc=sum(res)/repeticiones
        print("auc: {}  gini: {}".format(auc,auc*2-1))
        f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
        variables.update(auc-prev_auc)
        variables.printValues()
        prev_auc=auc
    variables.printValues()
    """
    
    """
    res=[]
    repeticiones=4
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.07,[1024, 512, 256]))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
    
        
    res=[]
    repeticiones=4
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.06,[1024, 512, 256, 64]))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))

    res=[]
    repeticiones=4
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.06,[2048,1024, 512, 256]))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
    
    res=[]
    repeticiones=4
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.1,[1024, 512, 256]))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
    
    res=[]
    repeticiones=4
    for i in range(repeticiones):
          res.append(train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data,FLAGS.just_test,0.04,[1024, 512, 256]))
    auc=sum(res)/repeticiones
    print("auc: {}  gini: {}".format(auc,auc*2-1))
    f.write('Media de las ultimas {} iteraciones: {} Gini:{}'.format(repeticiones,auc,auc*2-1))
    """
    f.close()

    
                 

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