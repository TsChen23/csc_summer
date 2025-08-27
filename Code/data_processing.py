#%%
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import time
import re
chunk_size = 1000000 

#分离出主合约
def seperate_main(df):
    date_set = df['date'].unique()
    res = pd.DataFrame(columns=df.columns)
    for date in date_set:
        temp = df[df['date'] == date]
        if date < 20240215:
            temp = temp[temp['securityid'] == 'TL2403']
        elif date < 20240515 and date >= 20240215:
            temp = temp[temp['securityid'] == 'TL2406']
        elif date < 20240815 and date >= 20240515:
            temp = temp[temp['securityid'] == 'TL2409']
        elif date < 20241115 and date >= 20240815:
            temp = temp[temp['securityid'] == 'TL2412']
        elif date < 20250215 and date >= 20241115:
            temp = temp[temp['securityid'] == 'TL2503']
        elif date < 20250515 and date >= 20250215:
            temp = temp[temp['securityid'] == 'TL2506']
        else:
            temp = temp[temp['securityid'] == 'TL2509']
        res = pd.concat([res,temp],axis=0)
    return res


#由于原数据中的日期是int格式的，需要转换成日期格式
def convert_int_to_time(timestamp):
    if len(timestamp) == 8:
        ts_str = '0' + timestamp
    else:
        ts_str = str(timestamp)
    
    hours = int(ts_str[0:2])
    minutes = int(ts_str[2:4])
    seconds = int(ts_str[4:6])
    milliseconds = int(ts_str[6:9])
    
    t = time(hour=hours, minute=minutes, second=seconds, microsecond=milliseconds*1000)
    
    return t

df = pd.DataFrame()

#原文件过大，需要划分成小块来进行处理
for chunk in pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\hq-cfl2-TL-2024\hq-cfl2-TL2403(EFP)-1-20250717174531884.csv", chunksize=chunk_size):
    temp = seperate_main(chunk)
    df = pd.concat([df,temp],axis=0)
for chunk in pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\hq-cfl2-TL-2024\hq-cfl2-TL-2025H1.csv", chunksize=chunk_size):
    temp = seperate_main(chunk)
    df = pd.concat([df,temp],axis=0)
df['time'] = df['time'].astype(int).astype(str)
df['time'] = df['time'].apply(convert_int_to_time)

#把非交易时间的数据剔除掉
starttime = time(hour=9,minute=30)
endtime = time(hour=15,minute=15)
df = df[(df['time']>=starttime) & (df['time']<=endtime)]

#保留指定列并把bid及ofr的price分离独自成列
print('已删除非交易数据')
column = ['securityid','date','time','preclose','open','high','low','last','offer_prices','offer_volumes','bid_prices','bid_volumes','total_volume_trade','total_value_trade','high_limited_price','low_limited_price','total_position','preopen_interest','presettle_price']
df = df[column]
df['bid_prices'] = df['bid_prices'].str.findall(r'\d+')
df[['Bid_Price1', 'Bid_Price2', 'Bid_Price3', 'Bid_Price4', 'Bid_Price5']] = df['bid_prices'].apply(pd.Series).astype(int)
print('已完成1')
df['offer_prices'] = df['offer_prices'].str.findall(r'\d+')
df[['Offer_Price1', 'Offer_Price2', 'Offer_Price3', 'Offer_Price4', 'Offer_Price5']] = df['offer_prices'].apply(pd.Series).astype(int)
print('已完成2')
df.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\分离bid及ofr后df.csv",index=False)
#%% 一次性运行完所有代码时间太长了，分两次运行
df = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\分离bid及ofr后df.csv")
df['bid_volumes'] = df['bid_volumes'].str.findall(r'\d+')
df[['Bid_Amount1', 'Bid_Amount2', 'Bid_Amount3', 'Bid_Amount4', 'Bid_Amount5']] = df['bid_volumes'].apply(pd.Series).astype(int)
print('已完成3')
df['offer_volumes'] = df['offer_volumes'].str.findall(r'\d+')
df[['Offer_Amount1', 'Offer_Amount2', 'Offer_Amount3', 'Offer_Amount4', 'Offer_Amount5']] = df['offer_volumes'].apply(pd.Series).astype(int)
df = df.drop(columns=['bid_prices','bid_volumes','offer_prices','offer_volumes'])
print(df)
df.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\分离bid及ofr后df.csv")
# %% resample成5min级别的（运行上方Run Cell可以直接运行
import pandas as pd
import numpy as np

#'time'这一列当小时为个位数时开头会有0，如09：30：00，把0去掉才能转换为datetime格式
def remove_leading_zero(s):
    if s.startswith('0'):
        return s[1:]  
    return s

df1 = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\分离bid及ofr后df.csv",index_col=0).drop(columns='Unnamed: 0')
df1['date'] = df1['date'].astype(str)
df1['time'] = df1['time'].astype(str).apply(remove_leading_zero)
df1['time'] = df1['date']+' '+df1['time']
df1['time_stamp'] = pd.to_datetime(df1['time'], format='mixed', errors='coerce')

#构建因子：交易方向
df1['trade_direction'] = 0

#如果最后成交价格比卖一价高，则认为是买方发起的交易，赋值为1，如果最后成交价格比买一价低，认为是卖方发起的交易，赋值为-1
df1.loc[df1['last'] > df1['Offer_Price1'], 'trade_direction'] = 1
df1.loc[df1['last'] < df1['Bid_Price1'], 'trade_direction'] = -1

mask = df1['trade_direction'] == 0

#如果在买一价与卖一价之间，则与上一最新成交价比较，如果大于则认为是买方发起的交易，赋值为1，赋值为-1的同理
df1.loc[mask & (df1['last'] >= df1['last'].shift(1)), 'trade_direction'] = 1
df1.loc[mask & (df1['last'] < df1['last'].shift(1)), 'trade_direction'] = -1

#将原始数据resample成10min级别的（需要先将时间戳列设定为index）
df1.set_index('time_stamp',inplace=True)
resampled_df = df1.resample('10T').agg({'preclose': 'last', 
                                       'open': 'last', 
                                       'high': 'last', 
                                       'low': 'last', 
                                       'last': 'last',
                                        'total_volume_trade': 'last', 
                                        'total_value_trade': 'last', 
                                        'high_limited_price': 'last',
                                        'low_limited_price': 'last', 
                                        'total_position': 'last', 
                                        'Bid_Price1':'last',
                                        'Bid_Price2':'last',
                                        'Bid_Price3':'last',
                                        'Bid_Price4':'last',
                                        'Bid_Price5':'last',
                                        'Offer_Price1':'last',
                                        'Offer_Price2':'last',
                                        'Offer_Price3':'last',
                                        'Offer_Price4':'last',
                                        'Offer_Price5':'last',
                                        'Bid_Amount1':'last',
                                        'Bid_Amount2':'last',
                                        'Bid_Amount3':'last',
                                        'Bid_Amount4':'last',
                                        'Bid_Amount5':'last',
                                        'Offer_Amount1':'last',
                                        'Offer_Amount2':'last',
                                        'Offer_Amount3':'last',
                                        'Offer_Amount4':'last',
                                        'Offer_Amount5':'last'})

#计算因子：交易不平衡（先计算每笔交易的量，然后再计算每10min中买方发起交易的量和卖方发起的交易的量的比值）
df1['cur_vol'] = df1['total_volume_trade'].diff()
df1[df1['cur_vol']<0]['cur_vol'] = df1['total_volume_trade']

unbalanced_vol = df1.groupby(pd.Grouper(freq='10min')).apply(lambda x: pd.Series({
    'buy_volume': x[x['trade_direction'] == 1]['cur_vol'].sum(),
    'sell_volume': x[x['trade_direction'] == -1]['cur_vol'].sum(),
    'volume_ratio': x[x['trade_direction'] == 1]['cur_vol'].sum() / x[x['trade_direction'] == -1]['cur_vol'].sum()
}))
resampled_df['unbalanced_vol'] = unbalanced_vol['volume_ratio']

max_value = resampled_df['unbalanced_vol'].replace([np.inf, -np.inf], np.nan).max()

#替换inf值
resampled_df['unbalanced_vol'].replace([np.inf, -np.inf], max_value, inplace=True)
resampled_df = resampled_df.dropna(how='all')

#计算每10min中第一分钟的价及量
resampled_df['amount_first_min'] = None
resampled_df['volume_first_min'] = None
for j in range(len(resampled_df)):
    start_time = resampled_df.index[j]
    #分离出第一分钟的数据
    first_min_data = df1[start_time:start_time + pd.Timedelta(minutes=1)]
    if not first_min_data.empty:
        #为每10分钟中第一分钟的总成交价及总量赋值
        resampled_df.at[start_time, 'amount_first_min'] = first_min_data['total_value_trade'].iloc[-1] 
        resampled_df.at[start_time, 'volume_first_min'] = first_min_data['total_volume_trade'].iloc[-1] 
        #计算第一分钟的平均成交价
        resampled_df['value_1min'] = np.where(
            resampled_df['amount_first_min'] - resampled_df['total_value_trade'].shift(1) > 0,
            resampled_df['amount_first_min'] - resampled_df['total_value_trade'].shift(1),
            resampled_df['amount_first_min']
            )
        resampled_df['volume_1min'] = np.where(
            resampled_df['volume_first_min'] - resampled_df['total_volume_trade'].shift(1) > 0,
            resampled_df['volume_first_min'] - resampled_df['total_volume_trade'].shift(1),
            resampled_df['volume_first_min']
            )
        resampled_df['price_first_min'] = (resampled_df['value_1min']/resampled_df['volume_1min'])
        #继续计算后14min的数据
        resampled_df['price_last14_min'] = (resampled_df['total_value_trade'] - resampled_df['amount_first_min'])/(resampled_df['total_volume_trade'] - resampled_df['volume_first_min'])
        resampled_df['volume_first_min1'] = resampled_df['volume_first_min'] - resampled_df['total_value_trade'].shift(1)
        resampled_df['amount_first_min1'] = resampled_df['amount_first_min'] - resampled_df['total_value_trade'].shift(1)

#计算benchprice，计算方法为反向价量加权
def calculate_benchprice(data):
    decay = 0.9
    data['bid_numerator'] = data['Bid_Price1']*data['Bid_Amount1'] + data['Bid_Price2']*data['Bid_Amount2']*decay + data['Bid_Price3']*data['Bid_Amount3']*decay*decay + data['Bid_Price4']*data['Bid_Amount4']*decay*decay*decay+ data['Bid_Price5']*data['Bid_Amount5']*decay*decay*decay*decay
    data['bid_denominator'] = data['Bid_Amount1'] + data['Bid_Amount2']*decay + data['Bid_Amount3']*decay*decay + data['Bid_Amount4']*decay*decay*decay + data['Bid_Amount5']*decay*decay*decay*decay
    data['offer_numerator'] = data['Offer_Price1']*data['Offer_Amount1'] + data['Offer_Price2']*data['Offer_Amount2']*decay + data['Offer_Price3']*data['Offer_Amount3']*decay*decay + data['Offer_Price4']*data['Offer_Amount4']*decay*decay*decay+ data['Offer_Price5']*data['Offer_Amount5']*decay*decay*decay*decay
    data['offer_denominator'] = data['Offer_Amount1'] + data['Offer_Amount2']*decay + data['Offer_Amount3']*decay*decay + data['Offer_Amount4']*decay*decay*decay + data['Offer_Amount5']*decay*decay*decay*decay
    data['bid_price'] = data['bid_numerator'] / data['bid_denominator']
    data['offer_price'] = data['offer_numerator'] / data['offer_denominator']
    data['bench_price'] = (data['bid_price'] * data['offer_denominator'] + data['offer_price'] * data['bid_denominator'])/ (data['offer_denominator'] + data['bid_denominator'])
    data = data.drop(columns = ['bid_numerator', 'bid_denominator', 'offer_numerator','offer_denominator', 'bid_price', 'offer_price'])
    return data

#创建时间戳变量，包括小时/分钟/秒等
def construct_timeindex(data):
    data['year'] = data['time_stamp'].dt.year
    data['month'] = data['time_stamp'].dt.month
    data['day'] = data['time_stamp'].dt.day
    data['hour'] = data['time_stamp'].dt.hour
    data['minute'] = data['time_stamp'].dt.minute
    data['second'] = data['time_stamp'].dt.second
    data['microsecond'] = data['time_stamp'].dt.microsecond
    print('时间变量创建完毕')
    return data 

resampled_df = calculate_benchprice(resampled_df)
resampled_df = resampled_df.reset_index(names='time_stamp')
resampled_df = construct_timeindex(resampled_df)
resampled_df.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\resample成5min后.csv",index=False)
#df1.set_index('time')
# %% 制作因子
import pandas as pd
import numpy as np
import math
from datetime import datetime

#将时间戳列转换为datetime格式
def trans_to_datetime(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

#获取两个时间戳之间相差多少秒，但是由于本方法是固定时间间隔，因此此函数已无意义
def get_seconds(x):
    return x.total_seconds()

#获取benchprice及lastprice之间的距离
def get_bl_dist(bp,lp):
    if lp==0:
        return np.nan
    else:
        return (bp-lp)/lp

#同上
def get_ml_dist(mp,lp):
    if lp==0:
        return np.nan
    else:
        return (mp-lp)/lp

#也是根据切片判断上个切片的交易主要是卖方发起的还是买方发起的，然后计算当前成交额与上个切片的挂单量的比值
def avgLastPrice_ratio(avgLastPrice, midPrice, CurrentTradeVol, depth_shift, askVol1_shift, askVol2_shift,askVol3_shift, askVol4_shift,askVol5_shift, 
                       bidVol1_shift,bidVol2_shift,bidVol3_shift,bidVol4_shift,bidVol5_shift):
    if avgLastPrice == midPrice:
        if depth_shift == 0:
            return CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/depth_shift
    elif avgLastPrice > midPrice:
        if askVol1_shift+askVol2_shift+askVol3_shift+askVol4_shift+askVol5_shift == 0:
            return  CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/ (askVol1_shift+askVol2_shift+askVol3_shift+askVol4_shift+askVol5_shift)
    else:
        if (bidVol1_shift+bidVol2_shift+bidVol3_shift+bidVol4_shift+bidVol5_shift) == 0:
            return CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/ (bidVol1_shift+bidVol2_shift+bidVol3_shift+bidVol4_shift+bidVol5_shift)

#计算高频偏度
def get_rskew(rol_sum_r3, rol_sum_r2):
    if math.isclose(rol_sum_r2, 0, abs_tol=1e-15) == True:
        return 0
    return (60 ** (1 / 2)) * rol_sum_r3 / (rol_sum_r2 ** (3 / 2))

#计算下行波动占比
def get_downward_ratio(rol_sum_r2_r_neg, rol_sum_r2):
    if math.isclose(rol_sum_r2, 0, abs_tol=1e-15) == True:
        return 0
    return rol_sum_r2_r_neg / rol_sum_r2

#去除价量相关性中的异常值（如空值，大于或者小于1的值等等）
def pvol_corr_adj(a):
    if np.isnan(a):
        return a
    elif (a > 1) or (a < -1):
        return np.nan
    else:
        return a

#对某列取sigmoid标准化
def get_sigmoid(a):
    if np.isnan(a):
        return a
    elif a == float("inf"):
        return a
    else:
        return 1/(1+math.exp(a))

#如果某列数值过大，则对ex做泰勒展开
def get_sigmoid_large(a):
    if np.isnan(a):
        return a
    elif a == float("inf"):
        return a
    else:
        return 1/(1+1+(a**1)/1+(a**2)/2+(a**3)/6+(a**4)/24+(a**5)/120)


df = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\resample成5min后.csv")
df['trade_time'] = df['time_stamp'].apply(lambda x: trans_to_datetime(x))
df = df.sort_values(by=["trade_time"], ascending=True, ignore_index=True)

#原数据价格尺度过大，需除以一万规范至三位数的量级，例如TL的价格将是110多
col_to_divide = ['preclose','open','high','low','last','total_value_trade','high_limited_price','low_limited_price','Bid_Price1','Bid_Price2','Bid_Price3','Bid_Price4','Bid_Price5','Offer_Price1','Offer_Price2','Offer_Price3','Offer_Price4','Offer_Price5','bench_price']
for col in col_to_divide:
    df[col] = df[col] / 10000
df['date'] = df['trade_time'].dt.date
day_list = df['date'].unique()

#此处flag用于最后合并每日创建的df
flag = 0

#开始计算因子
for d in day_list:
    data_temp = df[df['date'] == d]
    #midprice等于bid1和offer1的均值
    data_temp['midPrice'] = (data_temp['Bid_Price1'] + data_temp['Offer_Price1'])/2
    data_temp = data_temp.reset_index()
    
    #当前切片的成交额和成交量
    first_amt = data_temp.loc[0,'total_value_trade']
    first_vol = data_temp.loc[0,'total_volume_trade']
    data_temp['CurrentTradeAmt'] = data_temp['total_value_trade'].diff()
    data_temp['CurrentTradeVol'] = data_temp['total_volume_trade'].diff()
    data_temp.loc[0,'CurrentTradeAmt'] = first_amt
    data_temp.loc[0,'CurrentTradeVol'] = first_vol
    data_temp.loc[data_temp['CurrentTradeAmt']<0, 'CurrentTradeAmt'] = data_temp['total_value_trade']
    data_temp.loc[data_temp['CurrentTradeVol']<0, 'CurrentTradeVol'] = data_temp['total_volume_trade']
    data_temp = data_temp[data_temp['CurrentTradeVol']!=0]
    
    #当前切片最高价和最低价之间的价差与成交量的比值
    data_temp['high_low_vol'] = (data_temp['high'] - data_temp['low'])/data_temp['CurrentTradeVol']
    
    #切片内成交均价
    data_temp['AvgLastPrice'] = (data_temp['CurrentTradeAmt']/data_temp['CurrentTradeVol'])
    data_temp['AvgLastPrice_shift'] = data_temp['AvgLastPrice'].shift()
    data_temp['AvgLastPrice_diff_sigm'] = data_temp['AvgLastPrice'].diff().apply(get_sigmoid)
    data_temp['AvgLastPrice_diff_sigm_p1'] = data_temp['AvgLastPrice_diff_sigm'].shift(-1)

    #benchprice（计算方法见上方的函数）
    data_temp['benchPrice_sigm'] = data_temp['bench_price'].apply(get_sigmoid)
    data_temp['midPrice_sigm'] = data_temp['midPrice'].apply(get_sigmoid)
    
    #benchprice和lastprice之间的距离
    data_temp["bl_dist"] = data_temp.apply(lambda x: get_bl_dist(x["bench_price"], x["last"]), axis=1)
    data_temp['bl_dist_sigm'] = data_temp['bl_dist'].apply(get_sigmoid)
    
    #benchprice和avgprice之间的距离
    data_temp["b_avl_dist"] = (data_temp["bench_price"]-data_temp["AvgLastPrice"]) / data_temp["AvgLastPrice"]
    data_temp['b_avl_dist_sigm'] = data_temp['b_avl_dist'].apply(get_sigmoid)
    
    #lastprice和avgprice之间的距离
    data_temp["l_avl_dist"] = (data_temp["last"]-data_temp["AvgLastPrice"]) / data_temp["AvgLastPrice"]
    data_temp['l_avl_dist_sigm'] = data_temp['l_avl_dist'].apply(get_sigmoid)
    
    #midprice和lastprice之间的距离
    data_temp["ml_dist"] = data_temp.apply(lambda x: get_ml_dist(x["midPrice"], x["last"]), axis=1)
    data_temp['ml_dist_sigm'] = data_temp['ml_dist'].apply(get_sigmoid)
    
    #midprice和avgprice之间的距离
    data_temp["m_avl_dist"] = (data_temp["midPrice"]-data_temp["AvgLastPrice"]) / data_temp["AvgLastPrice"]
    data_temp['m_avl_dist_sigm'] = data_temp['m_avl_dist'].apply(get_sigmoid)
    
    #benchprice和avgprice之间的距离
    data_temp["ba_dist"] = (data_temp["bench_price"]-data_temp['AvgLastPrice']) / data_temp["bench_price"]
    data_temp['ba_dist_sigm'] = data_temp['ba_dist'].apply(get_sigmoid)
    
    #市场深度，也就是挂单量的加总
    data_temp["depth"] = (data_temp["Offer_Amount1"]+data_temp["Offer_Amount2"]+data_temp["Offer_Amount3"]+data_temp["Offer_Amount4"]+data_temp["Offer_Amount5"]+
                                  data_temp["Bid_Amount1"]+data_temp["Bid_Amount2"]+data_temp["Bid_Amount3"]+data_temp["Bid_Amount4"]+data_temp["Bid_Amount5"]) /2
    data_temp['depth_sigm'] = data_temp['depth'].apply(get_sigmoid_large)
    cols = ["Offer_Amount1", "Offer_Amount2","Offer_Amount3","Offer_Amount4","Offer_Amount5",
                    "Bid_Amount1", "Bid_Amount2","Bid_Amount3","Bid_Amount4","Bid_Amount5", "depth"]
    for col in cols:
        data_temp[col + "_shift"] = data_temp[col].shift()
    
    #当前切片成交量与上一切片市场深度的比值
    data_temp["tVol_depth"] = data_temp["CurrentTradeVol"]/ [m if m != 0 else 0.0000001 for m in data_temp["depth_shift"]]
    data_temp['tVol_depth_sigm'] = data_temp['tVol_depth'].apply(get_sigmoid_large)
    
    #详细计算方法见上方函数，简单来讲就是根据成交均价与中间价的关系判断当前切片的交易是由谁发起的，然后计算当前切片成交量与买方/卖方挂单量的比值
    data_temp["tVol_abVol"] = data_temp.apply(lambda x: avgLastPrice_ratio(x['AvgLastPrice'], x['AvgLastPrice_shift'],x["CurrentTradeVol"], x["depth_shift"] ,x["Offer_Amount1_shift"],x["Offer_Amount2_shift"],x["Offer_Amount3_shift"],x["Offer_Amount4_shift"],x["Offer_Amount5_shift"],
                        x["Bid_Amount1_shift"], x["Bid_Amount2_shift"], x["Bid_Amount3_shift"],  x["Bid_Amount4_shift"],  x["Bid_Amount5_shift"]),  axis=1)
    data_temp['tVol_abVol_sigm'] = data_temp['tVol_abVol'].apply(get_sigmoid_large)
    
    #计算每个切片的涨跌幅
    data_temp["Return"] = data_temp['AvgLastPrice'].pct_change()
    data_temp['Return_std'] = data_temp['Return'].rolling(window = 5, min_periods = 2).std()
    data_temp['Return_sigm'] = data_temp['Return'].apply(get_sigmoid)
    data_temp["Return_p_1"] = data_temp["Return"].shift(-2)
    data_temp['Return_p_1_sigm'] = data_temp['Return_p_1'].apply(get_sigmoid)
    
    #计算涨跌幅的2/3次方，及成交均价的shift等，为后续高频偏度计算做准备
    data_temp["Return3"] = data_temp["Return"] ** 3
    data_temp["Return2"] = data_temp["Return"] ** 2
    data_temp['ret_neg'] = data_temp["Return"] < 0
    data_temp['Return2_r_neg'] = data_temp['ret_neg'] * data_temp["Return2"]
    data_temp['AvgLastPrice_shift_ratio'] = data_temp['AvgLastPrice'] / data_temp['AvgLastPrice_shift']
    data_temp['AvgLastPrice_shift_ratio_sigm'] = data_temp['AvgLastPrice_shift_ratio'].apply(get_sigmoid)
    data_temp['CurrentTradeVol_sigm'] = data_temp['CurrentTradeVol'].apply(get_sigmoid_large)
    data_temp['CurrentTradeAmt_sigm'] = data_temp['CurrentTradeAmt'].apply(get_sigmoid_large)
    
    #新建一个df，因为要新设立index
    data_temp3 = data_temp
    data_temp3['index_org'] = data_temp3.index
    data_temp3['trade_time'] = pd.to_datetime(data_temp3['trade_time'])
    data_temp3 = data_temp3.set_index('trade_time', drop=False)

    data_temp3['rol_sum_r3'] = data_temp3["Return3"].rolling('120T').sum()
    data_temp3['rol_sum_r2'] = data_temp3["Return2"].rolling('120T').sum()
    data_temp3['rol_sum_r2_r_neg'] = data_temp3["Return2_r_neg"].rolling('120T').sum()
    data_temp3['ALP_ratio_prod'] = data_temp3['AvgLastPrice_shift_ratio'].rolling('120T').apply(np.nanprod, raw=True)

    #高频偏度
    data_temp3["rskew"] = data_temp3.apply(lambda x: get_rskew(x['rol_sum_r3'], x['rol_sum_r2']), axis=1)
    data_temp3['rskew'] = np.real(data_temp3['rskew'])
    data_temp3['rskew_sigm'] = data_temp3['rskew'].apply(get_sigmoid)
    #下行波动占比：涨跌幅为负数的平方之和比上所有涨跌幅的平方之和
    data_temp3["downward_ratio"] = data_temp3.apply(lambda x: get_downward_ratio(x['rol_sum_r2_r_neg'], x['rol_sum_r2']), axis=1)
    data_temp3['downward_ratio_sigm'] = data_temp3['downward_ratio'].apply(get_sigmoid)
    #改进反转
    data_temp3["reverse"] = data_temp3['ALP_ratio_prod'] - 1
    data_temp3['reverse_sigm'] = data_temp3['reverse'].apply(get_sigmoid)
    #量价相关性：成交均价与成交量的相关系数
    data_temp3["pvol_corr"] = data_temp3['AvgLastPrice'].rolling('120T').corr(data_temp3['CurrentTradeVol'],numeric_only=True)
    data_temp3["pvol_corr"] = data_temp3["pvol_corr"].apply(pvol_corr_adj)
    data_temp3['pvol_corr_sigm'] = data_temp3['pvol_corr'].apply(get_sigmoid)
    
    #将每一天的df拼接起来
    data_temp3 = data_temp3.set_index('index_org')
    if flag == 0:
        res = data_temp3
        flag += 1
    else:
        res = pd.concat([res, data_temp3], ignore_index=True)

#给价格的涨跌幅打标签
def label(x):
    if x > 0.02:
        return 2
    elif x >= -0.02 and x <= 0.02:
        return 1
    else:
        return 0

#drop掉一些无用的col，前面的两个是因为是日期，后面的三个存在未来信息，需要删去
#col_drop = ['trade_time','date','Return_p_1','Return_p_1_sigm','AvgLastPrice_diff_sigm_p1']
#for col in col_drop:
    #if col in res.columns:
        #res = res.drop(columns = [col])

#生成lag两阶的因子
for col in res.columns[1:-1]:
    res[col+'shift'] = res[col].shift(1)

#生成标签及均线
res['diff'] = res['AvgLastPrice'].diff()
res['label'] = res['diff'].apply(label).shift(-1)
res = res.fillna(method='ffill').dropna()
res['MA3'] = res['AvgLastPrice'].rolling(window=6).mean()
res['MA9'] = res['AvgLastPrice'].rolling(window=30).mean()
res['to_shortavg'] = res['AvgLastPrice']-res['MA3']
res['to_longavg'] = res['AvgLastPrice']-res['MA9']

#把label列移到最后
cols = res.columns.tolist()  
cols.remove('label')  
cols.append('label')  
res = res[cols]
print(res['label'].value_counts(normalize=True))

res.drop(columns = 'index').to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\计算因子后.csv",index=False)
# %%计算IC值
factor_list_sigm = ["benchPrice_sigm", "midPrice_sigm", "bl_dist_sigm", "b_avl_dist_sigm", "ml_dist_sigm", "m_avl_dist_sigm",
                    "ba_dist_sigm", "depth_sigm","tVol_depth_sigm", "tVol_abVol_sigm", "rskew_sigm", "downward_ratio_sigm", "reverse_sigm",
                    "pvol_corr_sigm",'AvgLastPrice_shift_ratio_sigm','CurrentTradeVol_sigm','CurrentTradeAmt_sigm','Return_sigm','l_avl_dist',
                    'unbalanced_vol','MA3','MA9','to_shortavg','to_longavg','year','month','day','hour','minute']
df = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\计算因子后.csv")
IClist = []
for factor in factor_list_sigm:
    ic = df[factor].corr(df['Return_p_1_sigm'])
    IClist.append(ic)
ic_res = pd.DataFrame({'因子名':factor_list_sigm,'IC值':IClist})
ic_res.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\因子一阶IC结果.csv")

