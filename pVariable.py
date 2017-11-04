# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:12:14 2017

@author: gmase
"""
import tensorflow as tf

class pVariable:
    def __init__(self,name):
        self.name=name
        self.setType(name)
        self.setParticle(name)
        
    def setType(self,name):
        if 'bin' in name:
            self.tipo='bin'
        elif 'cat' in name:
            self.tipo='cat'
            self.setBuckets()
        else:
            self.tipo='cont'
    def setBuckets(self):
        self.buckets=8
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
            return tf.feature_column.categorical_column_with_identity( self.name, num_buckets=self.buckets,default_value=self.buckets-1)
        else:
            return tf.feature_column.numeric_column(self.name)

                
                
                
                
                
        