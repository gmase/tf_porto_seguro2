

import argparse
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
#from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

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

#bucketsOf=[5,2,7,12,2,10,2,18,2,2,2,5,3,105]
bucketsOf=[6,3,8,13,3,11,3,19,3,3,3,6,4,106]
bucketsOf=[i+2 for i in bucketsOf]

binBuckets=[3 for i in range(17)]

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


    
FLAGS = None




def datetime(x):
    return np.array(x, dtype=np.datetime64)

def pruebaStocks():   
    aapl = np.array(AAPL['adj_close'])
    aapl_dates = np.array(AAPL['date'], dtype=np.datetime64)
    
    window_size = 30
    window = np.ones(window_size)/float(window_size)
    aapl_avg = np.convolve(aapl, window, 'same')
    
    p2 = figure(x_axis_type="datetime", title="AAPL One-Month Average")
    p2.grid.grid_line_alpha = 0
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Price'
    p2.ygrid.band_fill_color = "olive"
    p2.ygrid.band_fill_alpha = 0.1
    
    p2.circle(aapl_dates, aapl, size=4, legend='close',
              color='darkgrey', alpha=0.2)
    
    p2.line(aapl_dates, aapl_avg, legend='avg', color='navy')
    p2.legend.location = "top_left"
    
    output_file("PortoSeguroStats.html", title="stocks.py example")
    
    show(gridplot([[p2]], plot_width=400, plot_height=400))  # open a browser


    
def pintaGrafica(trainSet):
    reporta=trainSet['reporta']
    total=reporta.count()
    msk=reporta==1
    fullGrid=list()
    
    for field in CSV_COLUMNS[2:]:
        campoEstudio=trainSet[field]
        campoEstudioPositives=campoEstudio[msk]
        
        #window_size = 30
        #window = np.ones(window_size)/float(window_size)
        #aapl_avg = np.convolve(campoEstudio, window, 'same')
        
        p1 = figure(x_axis_type="linear", title=field)
        p1.grid.grid_line_alpha = 0
        p1.xaxis.axis_label = 'Field values'
        p1.yaxis.axis_label = 'Value repetitions'
        p1.ygrid.band_fill_color = "olive"
        p1.ygrid.band_fill_alpha = 0.1
        
        p2 = figure(x_axis_type="linear", title=field)
        p2.grid.grid_line_alpha = 0
        p2.xaxis.axis_label = 'Field values'
        p2.yaxis.axis_label = 'Positives %'
        p2.ygrid.band_fill_color = "olive"
        p2.ygrid.band_fill_alpha = 0.1
        
        hist, edges = np.histogram(campoEstudio, density=False, bins=10)
        p1.quad(top=hist*100/total, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="#036564", line_color="#033649")
            
        hist2, edges2 = np.histogram(campoEstudioPositives, density=False, bins=10)
        hist2=hist2*100.0/hist
        p2.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
            fill_color="#036564", line_color="#033649")
            
        p2.legend.location = "top_left"
        fullGrid.append([p1,p2])
    
    
    output_file("PortoSeguroStats.html", title="stats.py")
    show(gridplot(fullGrid, plot_width=400, plot_height=400))  # open a browser
    #show(gridplot([[p1,p2]], plot_width=400, plot_height=400))  # open a browser

def main():

    #pruebaStocks()
    
    trainSet = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    ind_05_classes = trainSet['ps_ind_05_cat'].unique()
    pintaGrafica(trainSet)
    """
    trainSet=pd.read_csv(filepath_or_buffer='train.csv',sep=',', delimiter=None, header='infer')
    testSet=pd.read_csv(filepath_or_buffer='test.csv',sep=',', delimiter=None, header='infer')
    print(trainSet.columns)
    ind_05_classes = trainSet['ps_ind_05_cat'].unique()
    n_classes = len(ind_05_classes) + 1
    print('Ind 05 has next classes: ', ind_05_classes)
    X=trainSet[['ps_ind_05_cat']]

    sys.exit("Quieto parao")
    """
    

    
                 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
