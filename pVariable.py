# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:12:14 2017

@author: gmase
"""
import tensorflow as tf
import numpy as np

class pVariables:
    def __init__(self,names,data,exclude):
        """
        excludeList=['ps_ind_02_cat',
        'ps_car_06_cat',
        'ps_car_12',
        'ps_calc_02',
        'ps_calc_03',
        'ps_calc_04',
        'ps_calc_05',
        'ps_calc_07',
        'ps_ind_02_cat',
        'ps_car_06_cat',
        'ps_car_12',
        'ps_calc_02',
        'ps_calc_03',
        'ps_calc_04',
        'ps_calc_05',
        'ps_calc_07']
        """
        
        excludeList=[
        'ps_calc_02',
        'ps_calc_03',
        'ps_calc_04',
        'ps_calc_05',
        'ps_calc_07',
        'ps_calc_02',
        'ps_calc_03',
        'ps_calc_04',
        'ps_calc_05',
        'ps_calc_07']
        if not exclude:
            excludeList=[]
        
        self.vars=[pVariable(i,data) for i in list(set(names) - set(excludeList))]
            
    def getSubset(self,prob):
        self.msk = np.random.rand(len(self.vars)) < prob
        variables_in=[]
        for i,v in enumerate(self.msk):
            if v>0:
                variables_in.append(self.vars[i])
                
        #m = np.ma.masked_where(msk>0, y) 
        """
        print(msk)
        self.variables_in = self.vars[msk]
        self.variables_out = self.vars[~msk]
        """
        return variables_in
    
        
    def update(self,improvement):
        if improvement>0.0:
            for i,v in enumerate(self.vars):
                if self.msk[i]>0:
                    v.value+=0.1
                else:
                    v.value-=0.1
        else:
            for i,v in enumerate(self.vars):
                if self.msk[i]>0:
                    v.value-=0.1
                else:
                    v.value+=0.1
                    
    def getCategoricals(self):
        toReturn=[]
        for i in self.vars:
            if i.tipo=='cat':
                toReturn.append(i.name)
        return toReturn
                
    def printValues(self):
        for i in self.vars:
            print("{}::{}").format(i.name,i.value)
        
            
class pVariable:
    def __init__(self,name,data):
        self.name=name
        self.setType(name,data)
        self.setParticle(name)
        self.value=0

        
    def setType(self,name,data):
        if 'bin' in name:
            self.tipo='bin'
        elif 'cat' in name:
            self.tipo='cat'
            self.setBuckets(data)
        else:
            self.tipo='cont'
    def setBuckets(self,data):
        self.buckets=len(np.bincount(data[self.name]+1))+1
        #self.buckets=20
    def setParticle(self,name):
        if 'car' in name:
            self.particle='car'
        elif 'ind' in name:
            self.particle='ind'
        elif 'reg' in name:
            self.particle='reg'
        else:
            self.particle='calc'
            
    def getTfVariable(self):
        if self.tipo=='bin':
            return tf.feature_column.numeric_column(self.name)
        elif self.tipo=='cat':
            return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity( self.name, num_buckets=self.buckets,default_value=self.buckets-1))
        else:
            return tf.feature_column.numeric_column(self.name)

                
                
                
                
                
        