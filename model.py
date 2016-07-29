# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:07:13 2016

@author: lvjia
"""
import os
from functools import reduce 

from WindPy import *
from datetime import *
import pandas as pd
import seaborn as sn
import numpy as np
import xlwt
import statsmodels.api as sm
from sklearn import linear_model

from database import *

path = 'C:\\Users\\lvjia\\Google 云端硬盘\\研一暑假\\bd quant research\\黑色商品产业链\\行业数据\\database'

os.listdir(path)
w.start()

data = read_data(path)