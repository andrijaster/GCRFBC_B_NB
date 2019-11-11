# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:42:49 2018

@author: Andrija Master
"""
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.utils import shuffle


data = arff.loadarff('emotions.arff')
df = pd.DataFrame(data[0])
mapping = {b'0': 0, b'1': 1}
df.replace(mapping, inplace = True)
df = shuffle(df)

atribute = df.iloc[:,:-6]
output = df.iloc[:,-6:]