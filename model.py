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
from sklearn import svm

path = 'C:\\Users\\lvjia\\Google 云端硬盘\\研一暑假\\bd quant research\\黑色商品产业链\\行业数据\\database'

os.listdir(path)
w.start()

profit = pd.read_csv(path+'\\return_RB_J_I.csv')
profit = profit.set_index('Unnamed: 0')
profit.index = (list(map(lambda x: datetime.strptime(x[0:10],'%Y-%m-%d'),profit.index)))
profit = profit.resample('w',how='last')



data_weekly = pd.read_csv(path+'\\data\\weekly_data.csv',encoding='GBK')
data_weekly = data_weekly.set_index('Unnamed: 0')
data_weekly.index = (list(map(lambda x: datetime.strptime(x,'%Y-%m-%d'),data_weekly.index)))
data_weekly['ret2'] = profit.loc[:,'RB_close'].pct_change()
# data_weekly['profit'] = profit.loc[:,'RB_close']

data_weekly['ret'] = data_weekly.loc[:,'螺纹钢_HRB400_20mm_价格_全国'].pct_change()
data_weekly['profit'] = data_weekly.loc[:,'螺纹钢_HRB400_20mm_价格_全国']


vars = ['房屋新开工面积_累计同比','商品房销售面积_累计同比','全国盈利钢厂',
        '社会总库存','螺纹钢毛利率','重点钢材企业库存',
        '全国高炉开工率','金融机构各项贷款余额_同比','ret','profit','螺纹钢_HRB400_20mm_价格_全国','ret2']
model_data = data_weekly.loc[:,vars]

model_data = model_data.ffill()


model_data['全国盈利钢厂_变化'] = model_data['全国盈利钢厂'] - model_data['全国盈利钢厂'].shift(1)

model_data['x1'] = model_data['房屋新开工面积_累计同比'].shift(8)
model_data['x2'] = model_data['商品房销售面积_累计同比'].shift(50)
model_data['x3'] = model_data['全国盈利钢厂_变化'].shift(4)
model_data['x4'] = model_data['螺纹钢毛利率'].shift(1)
model_data['x5'] = model_data['重点钢材企业库存'].pct_change(periods=52).shift(1)
model_data['x6'] = model_data['全国高炉开工率'].shift(2)
model_data['x7'] = model_data['金融机构各项贷款余额_同比'].shift(3)
model_data['x8'] = model_data['x4']*model_data['x4']
model_data['x9'] = model_data['全国盈利钢厂'].shift(1)
model_data['y'] = model_data['ret'].shift(-1)

# model_data = model_data['2007-12-30':]
model_data['y2'] = model_data['ret2'].shift(-1)

model_data['y2'] = model_data['y2'].fillna(0)
model_data['ret2'] = model_data['ret2'].fillna(0)

model_data = model_data.dropna()


# 模型
data_w1 = model_data['2007-01-01':'2015-01-01']
data_w2 = model_data['2015-01-01':'2016-12-30']

y1 = data_w1.loc[:, 'y']
x = ['x1','x2','x3','x4','x5','x6','x7','x8','x9']
X1 = data_w1[x]

y2 = data_w2.loc[:, 'y']
X2 = data_w2[x]

X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)

model_name = 'ols'


if model_name == 'ols':
	model = sm.OLS(y1, X1)
	result = model.fit()
	print(result.summary())

elif model_name == 'ridge':
	model = linear_model.Ridge(alpha=0.1)
	result = model.fit(y=y1, X=X1)
	print(result.coef_)

elif model_name == 'lasso':
	model = linear_model.Lasso(alpha=0.1)
	result = model.fit(y=y1, X=X1)
	print(result.coef_)

elif model_name == 'SVR':
	# bad
	model = svm.SVR(C=0.0001)
	result = model.fit(y=y1, X=X1)
# print(result.coef_)

else:
	pass

y_hat1 = np.array(result.predict(X1))
y_indicator1 = np.where(y_hat1>0,1,-1)
ret1 = y1*y_indicator1
(y1*y_indicator1).cumsum().plot()
print(y_indicator1)
print(np.sqrt(52)*ret1.mean()/ret1.std())


# fig = plt.figure(figsize = (15,7) )
# # subplot(311)
y_hat2 = np.array(result.predict(X2))
y_indicator2 = np.where(y_hat2>0,1,-1)
ret2 = y2*y_indicator2
(y2*y_indicator2).cumsum().plot()
print(y_indicator2)
print(np.sqrt(52)*ret2.mean()/ret2.std())




# subplot(211)
#
# (y2*y_indicator2).plot()
#
#
# subplot(212)
# plt.plot(y2.index,y_hat2)
# print(sqrt(52)*ret2.mean()/ret2.std())
