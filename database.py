# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import os
from functools import reduce

from WindPy import *
from datetime import *
import pandas as pd
import seaborn as sn
import numpy as np
import xlwt


def transform(data, column_names):
    df = pd.DataFrame(data.Data, columns=data.Times, index=column_names).T
    df.index = map(lambda x: x.strftime('%Y-%m-%d'), df.index)
    return df


def initialize(start='2007-01-01', end=date.today(), csv="code2.csv"):
    # 获取变量列表
    codes = pd.read_csv(path + '\\' + csv, encoding='GBK').dropna()
    variables_id = reduce(lambda x, y: str(x) + ',' + str(y), codes['id'])
    variables_zh = reduce(lambda x, y: str(x) + ',' + str(y), codes['name']).split(sep=',')

    # 寻找旧数据
    if 'raw_data.csv' in os.listdir(path):
        # 更新数据
        raw_data_old = pd.read_csv(path + '\\raw_data.csv', encoding='utf-8')
        raw_data_old = raw_data_old.set_index('Unnamed: 0')
        last_update_day = raw_data_old.index[-1]
        # raw_data_old.index = map(lambda x: datetime.strptime(str(x)[:10], '%Y-%m-%d'), raw_data_old.index)
        today = date.today().strftime('%Y-%m-%d')

        data_temp = w.edb(variables_id, last_update_day, today, '')
        raw_data_new = transform(data_temp, variables_zh)
        # 删去新数据中的旧数据部分
        raw_data_new = raw_data_new[last_update_day:today]
        # 删去旧数据中的最后一天
        raw_data_old = raw_data_old.iloc[:-1, :]

        df = pd.concat([raw_data_old, raw_data_new])
        df.to_csv(path + '\\raw_data.csv', encoding='utf-8')


    else:
        # 重建数据
        date_list = w.tdays(start, end, "Days=Weekdays").Data[0]
        # 初始化DataFrame
        data = pd.DataFrame(index=w.tdays(start, end, "Days=Weekdays").Data)

        data_temp = w.edb(variables_id, start, end, '')
        df = transform(data_temp, variables_zh)
        df.to_csv(path + '\\raw_data.csv', encoding='utf-8')

    df.index = map(lambda x: datetime.strptime(x, '%Y-%m-%d'), df.index)

    # for idx, code_group in code_groups:
    for idx in range(1, 13):
        # variables_id = reduce( lambda x,y: str(x)+','+str(y),code_group['id'])
        # variables_zh = reduce( lambda x,y: str(x)+','+str(y),code_group['name']).split(sep=',')
        # data_temp = w.edb(variables_id, start, end,'')
        # df = transform(data_temp, variables_zh)

        if idx == 1:
            df = data_update1(df)

        elif idx == 2:
            df = data_update2(df)

        elif idx == 3:
            df = data_update3(df)
            print(df)
        elif idx == 4:
            df = data_update4(df)

        elif idx == 6:
            df = data_update6(df)

        elif idx == 7:
            df = data_update7(df)

        elif idx == 8:
            df = data_update8(df)

        elif idx == 9:
            df = data_update9(df)

        elif idx == 10:
            df = data_update10(df)

        elif idx == 11:
            df = data_update11(df)

        elif idx == 12:
            df = data_update12(df)

            # df.to_csv(path+'\\'+str(idx)+'.csv')
        # data[df.columns] = df

        data_daily = df.ffill()
        data_daily.to_csv(path + '\\data\\daily_data.csv', encoding='utf-8')
        data_weekly = df.resample('w', how='last').ffill()
        data_weekly.to_csv(path + '\\data\\weekly_data.csv', encoding='utf-8')
        data_monthly = df.resample('m', how='last').ffill()
        data_monthly.to_csv(path + '\\data\\monthly_data.csv', encoding='utf-8')

    return df


def data_update1(df):
    # 月度
    # 核心CPI 2013年1月
    # 固定投资完成额 缺1月数据
    # 金融机构各项贷款余额_新增 缺1月数据
    # 外汇占款余额 缺16年数据

    # ["CPI_同比","CPI食品_同比","CPI非食品_同比","核心cpi",
    #         'PMI','PMI新订单','PMI采购量','M2','M2_同比',
    #         '金融机构各项贷款余额','金融机构各项贷款余额_同比',
    #         '外汇占款余额','固定资产投资完成额']
    # 计算同比新增量
    df['金融机构各项贷款余额_新增'] = df['金融机构各项贷款余额'] - df['金融机构各项贷款余额'].shift(1)
    df['金融机构各项贷款余额_新增_同比'] = df['金融机构各项贷款余额_新增'] / df['金融机构各项贷款余额_新增'].shift(12) - 1.0
    return df


def data_update2(df):
    # 日度
    # MB铁矿石价格指数 2011年1月
    # 废钢总 2007年4月
    # 焦煤 焦炭 2007年1月
    # ["唐山二级冶金焦_含税", "张家港废钢_不含税", "秦皇岛港_大同优混", "热轧卷板_3.0mm_上海",
    #         '冷轧卷板_1.0mm_上海','螺纹钢_HRB400mm_上海','中板_普20mm_上海',
    #         'TSI铁矿石价格指数_62%','TSI铁矿石价格指数_58%','MB铁矿石价格指数','硅锰FeMnSi17_南方',"张家港废钢_含税"]
    # 获取外汇数据
    raw_fx = w.wsd("USDCNY.EX", "close", "2001-01-01", date.today(), "Period=d")
    fx = pd.DataFrame(raw_fx.Data, columns=raw_fx.Times).T
    df['人民币美元中间价'] = fx
    # 整合废钢价格数据
    df['张家港废钢_含税总'] = np.where(df['张家港废钢_含税'].isnull(), df['张家港废钢_不含税'] * 1.17, df['张家港废钢_含税'])
    df = df.fillna(method='ffill')

    # 参数输入
    n1 = 1.6  # 铁矿石
    n2 = 0.5  # 焦炭
    n3 = 0.1  # 焦煤
    cost1 = 220
    n4 = 0.96
    n5 = 0.1  # 废钢
    n6 = 0.02
    cost2 = 220
    tax = 0.17
    c1 = 300
    c2 = 700
    c3 = 120
    c4 = 450

    # 计算生铁成本
    df['铁矿石到厂价'] = df['MB铁矿石价格指数'] * df['人民币美元中间价']
    df['生铁成本'] = n1 * df['铁矿石到厂价'] + df['唐山二级冶金焦_含税'] * n2 / (1 + tax) + df['秦皇岛港_大同优混'] * n3 / (1 + tax) + cost1

    # 计算钢坯成本
    df['钢坯成本'] = df['生铁成本'] * n4 + df['张家港废钢_含税总'] / (1 + tax) * n5 + cost2 + df['硅锰FeMnSi17_南方'] * n6 / (1 + tax)

    # 计算钢材成本
    df['热轧成本'] = df['钢坯成本'] + c1
    df['冷轧成本'] = df['热轧成本'] + c2
    df['螺纹钢成本'] = df['钢坯成本'] + c3
    df['中厚板成本'] = df['钢坯成本'] + c4

    # 计算毛利率
    df['热轧毛利率'] = (df['热轧卷板_3.0mm_上海'] / (1 + tax)) / (df['热轧成本']) - 1
    df['冷轧毛利率'] = (df['冷轧卷板_1.5mm_价格_上海'] / (1 + tax)) / (df['冷轧成本']) - 1
    df['螺纹钢毛利率'] = (df['螺纹钢_HRB400_20mm_价格_上海'] / (1 + tax)) / (df['螺纹钢成本']) - 1
    df['中厚板毛利率'] = (df['中板_普20mm_价格_上海'] / (1 + tax)) / (df['中厚板成本']) - 1

    return df


def data_update3(df):
    # 月度
    # 缺每年一月数据
    # 价格指数 2010年6月
    # ["国房景气指数", "房地产开发投资完成额_累计值", "房地产开发投资完成额_累计同比", "房屋施工面积_累计值",
    #            '房屋施工面积_累计值_同比','房屋新开工面积_累计值','房屋新开工面积_累计同比',
    #            '商品房销售面积_累计值','商品房销售面积_累计同比','样本住宅平均价格','百城住宅价格指数_环比',"百城住宅价格指数_同比"]
    # df = df.resample('m')
    # 填充1月数据(暂不填充)
    # for i in range(len(df)):
    #    if df.index[i].month == 1:
    #        for j in [1,3,5,7]:
    #            df.iloc[i,j] = df.iloc[i+1,j]/2
    #        for j in [2,4,6,8]:
    #            df.iloc[i,j] = df.iloc[i+1,j]
    return df


def data_update4(df):
    # 月度
    # 缺每年一月数据
    # 固定资产投资完成额_国家铁路 缺失较多 16年4、5、6月
    # ["固定资产投资完成额_累计值", "固定资产投资完成额_累计同比",
    #         "固定资产投资完成额_国家铁路_累计值","固定资产投资完成额_国家铁路_累计同比",
    #         '固定资产投资完成额_基础设施建设投资_累计值','固定资产投资完成额_基础设施建设投资_累计同比',
    #         '新增固定资产投资完成额_累计值','新增固定资产投资完成额_累计同比']

    return df


def data_update5(df):
    # ['全国粗钢表观消费量','全国钢材表观消费量']
    # 缺2016年1、2月数据
    # 计算同比
    df['粗钢表观消费量_同比'] = df['全国粗钢表观消费量'] / df['全国粗钢表观消费量'].shift(12) - 1.0
    df['粗钢表观消费量_环比'] = df['全国粗钢表观消费量'] / df['全国粗钢表观消费量'].shift(1) - 1.0
    df['钢材表观消费量_同比'] = df['全国钢材表观消费量'] / df['全国钢材表观消费量'].shift(12) - 1.0
    df['钢材表观消费量_环比'] = df['全国钢材表观消费量'] / df['全国钢材表观消费量'].shift(1) - 1.0

    return df


def data_update6(df):
    # ['螺纹钢_期货收盘价_活跃','螺纹钢_期货成交量_活跃','螺纹钢_期货持仓量_活跃',
    #        '热轧_期货收盘价_活跃','焦煤_期货收盘价_活跃','焦炭_期货收盘价_活跃','线材_期货收盘价_活跃','铁矿石_期货收盘价_活跃' ]
    # 计算涨跌幅
    # 铁矿石 13年10月
    # 焦炭   11年4月
    df['螺纹涨幅'] = df['螺纹钢_期货收盘价_活跃'].pct_change()
    df['热轧涨幅'] = df['热轧_期货收盘价_活跃'].pct_change()
    df['焦煤涨幅'] = df['焦煤_期货收盘价_活跃'].pct_change()
    df['线材涨幅'] = df['焦炭_期货收盘价_活跃'].pct_change()
    df['铁矿石涨幅'] = df['铁矿石_期货收盘价_活跃'].pct_change()
    # 计算盘面利润
    df['盘面利润'] = ((df['螺纹钢_期货收盘价_活跃'] - 1.6 * df['铁矿石_期货收盘价_活跃'] - 0.5 * df['焦炭_期货收盘价_活跃'] - 900))

    return df


def data_update7(df):
    # 螺纹钢_全国 2009年1月
    # 热轧_全国 2011年1月
    # 冷轧_全国 2007年1月
    # 更新时间 ：晚上
    # 更新频率：每天
    # ['螺纹钢_HRB400_20mm_价格_全国','螺纹钢_HRB400_20mm_价格_天津','螺纹钢_HRB400_20mm_价格_广州','螺纹钢_HRB400_20mm_价格_上海','螺纹钢_HRB400_20mm_价格_北京',
    #         '热轧卷板_4.75mm_价格_全国','热轧卷板_4.75mm_价格_天津','热轧卷板_4.75mm_价格_广州','热轧卷板_4.75mm_价格_上海',
    #         '热轧卷板_4.75mm_价格_北京',
    #         '冷轧卷板_1.5mm_价格_天津','冷轧卷板_1.5mm_价格_广州','冷轧卷板_1.5mm_价格_上海','冷轧卷板_1.5mm_价格_北京',
    #         '中板_普20mm_价格_天津','中板_普20mm_价格_广州','中板_普20mm_价格_上海','中板_普20mm_价格_北京',
    #         '钢坯__价格_唐山'
    #        ]
    # 计算价差
    df['钢坯、螺纹价差'] = df['螺纹钢_HRB400_20mm_价格_全国'] - df['钢坯__价格_唐山']
    # 计算全国平均价
    df['钢坯、热轧价差'] = df.iloc[:, 6:10].mean(axis=1) - df['钢坯__价格_唐山']
    df['钢坯、冷轧价差'] = df.iloc[:, 10:14].mean(axis=1) - df['钢坯__价格_唐山']

    return df


def data_update8(df):
    # 五大钢材 2007年1月 周度
    # 重点钢材企业库存 2009年5月  月度

    # ['螺纹钢_库存','线材_库存','热卷板_库存','中板_库存','冷轧_库存','重点钢材企业库存']
    df['社会总库存'] = df.iloc[:, 0:5].sum(axis=1)

    return df


def data_update9(df):
    # 2012年8月 周数据
    # ['全国高炉开工率','全国高炉检修容积','全国高炉检修量',
    #        '全国盈利钢厂','全国检修钢厂','唐山钢厂高炉开工率','唐山钢厂产能利用率' ]


    return df


def data_update10(df):
    # 粗钢产能、粗钢产能利用率、淘汰落后产能炼钢任务完成量 2008年起 年度
    # 各地钢厂产能   周度

    # ['粗钢产能','粗钢产能利用率','淘汰落后产能_炼钢_任务完成量',
    #        '螺纹钢_主要钢厂产能_华东','螺纹钢_主要钢厂产能_南方',
    # '螺纹钢_主要钢厂产能_北方','螺纹钢_主要钢厂产能_全国' ]
    return df


def data_update11(df):
    # 钢材产量 2007年1月 周度
    # ['钢材产量_当月值','钢材产量_当月同比','钢材产量_累计值',
    #        '钢材产量_累计同比','钢材产量_重点企业']
    return df


def data_update12(df):
    # 进出口数据 2007年1月 月度
    # ['出口数量_钢材_累计值','出口数量_钢材_累计同比','出口数量_钢材_当月值','出口平均单价_钢材_当月值',
    #        '进口数量_钢材_累计值','进口数量_钢材_累计同比','进口数量_钢材_当月值','进口平均单价_钢材_当月值']

    df['净出口数量_钢材_累计值'] = df['出口数量_钢材_累计值'] - df['进口数量_钢材_累计值']
    df['净出口数量_钢材_当月值'] = df['出口数量_钢材_当月值'] - df['进口数量_钢材_当月值']
    return df


def read_data(path):
    data = pd.read_csv(path + '\\check.csv', encoding='GBK')
    return data


# path = 'C:\\Users\\lvjia\\Google 云端硬盘\\研一暑假\\bd quant research\\黑色商品产业链\\行业数据\\database'
#
# os.listdir(path)
# w.start()
# csv = 'code.csv'
#
# data = read_data(path)
# data.to_csv(path+ '\\check.csv')




if __name__ == '__main__':
    path = 'C:\\Users\\lvjia\\Google 云端硬盘\\研一暑假\\bd quant research\\黑色商品产业链\\行业数据\\database'
    dd = initialize(start='2007-07-27', end=date.today(), csv="code.csv")



