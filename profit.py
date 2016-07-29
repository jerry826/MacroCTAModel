# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:47:34 2016

@author: lvjia
"""

from WindPy import *
from datetime import *
import pandas as pd
import seaborn as sn
import numpy as np


# 获取螺纹钢期货的所有合约数据,开始于2014年1月1日,合约从1403开始
def get_contracts(start_date = '2014-01-01',start = '1403'):
    
    
    value_frames = []
    close_frames = []
    for i in range(14,18,1):
        for j in range(1,13):
            # 生成合约代码
            str1 = str(i)
            str2 = str(j)
            if len(str1) == 1:
                str1 = '0'+str1        
            
            if len(str2) == 1:
                str2 = '0'+str2
            sid = 'RB'+str1+str2+'.SHF'
    
            data = w.wsd(sid, "close,volume", start_date, date.today(), "")
            df = pd.DataFrame(data.Data,columns=data.Times,index=data.Fields).T
            # 成交量
            df1 = pd.DataFrame(df['VOLUME'])
            df1.columns=[str1+str2]
            value_frames.append(df1)
            # 收盘价
    #         df2 = pd.DataFrame(df['CLOSE'])
    #         df2.columns=[str1+str2]
    #         close_frames.append(df2)
               
    value_result = pd.concat(value_frames,axis=1)
    return value_result
# close_result = pd.concat(close_frames,axis=1)
# return_result= close_result.pct_change()

def choose_contracts(value_result):

    contracts = [] 
    for i in range(0,len(value_result)):
        contracts.append((np.argmax(value_result.iloc[i,:])))
    contracts = pd.DataFrame(contracts,index = value_result.index)
    contracts = contracts.dropna()
    return contracts


def change_contract(contract):
    if int(contract[-1]) == 10:
        contract = contract[:2]+'09'
    return contract


def get_data(contracts):
    # 获取每天螺纹钢主力合约对应的螺纹钢、铁矿石和焦炭价格
    v_list = []
    index_list = []
    for k in range(0,len(contracts),1):
        contract = contracts.iloc[k,0]
        time = (contracts.index[k])
        time = str(time)[0:4]+str(time)[5:7]+str(time)[8:10]
        RB =  "RB"+contract+'.SHF'        
        contract1 = change_contract(contract)
        J = 'J'+contract1+'.DCE'
        I = 'I'+contract1+'.DCE'
        data = w.wss(RB+','+J+','+I, "pre_close,close","tradeDate="+time+";priceAdj=U;cycle=D")
        v = []
        for i in range(3):
            v.append((data.Data[1][i]-data.Data[0][i])/data.Data[1][i])
        v = v + data.Data[0]+data.Data[1]
        v_list.append(v)
        index_list.append(contracts.index[k])
    
    columns = ['RB','J','I','RB_pre_close','J_pre_close','I_pre_close','RB_close','J_close','I_close']
    data = pd.DataFrame(v_list,index=index_list,columns=columns)
    data['contract'] = contracts
    # rr.to_csv('return_RB_J_I.csv')
    
    return data

def calculate(rr):
    rr['profit'] = rr['RB_close']/1.17-1.6*rr['I_close']/1.17-0.55*rr['J_close']/1.17
    # rb/1.17-(1.6*i/1.17+0.55*j/1.17+560)
    return rr



def get_profit(start_date = '2014-01-01',start = '1403',path='C:\\'):
    value_result = get_contracts(start_date = start_date,start = start)
    contracts = choose_contracts(value_result)
    rr = get_data(contracts)
    rr = calculate(rr)
    rr.to_csv(path + '\\return_RB_J_I.csv',encoding='utf-8')
    return rr

def test(start_date = '2014-01-01',start = '1403',path=''):
    rr = get_profit(start_date, start, path)
    return rr

if __name__ == '__main__':
    path = 'C:\\Users\\lvjia\\Google 云端硬盘\\研一暑假\\bd quant research\\黑色商品产业链\\行业数据\\database'
    rr = test(start_date = '2014-01-01',start = '1403', path=path)





