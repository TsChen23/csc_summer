#%%
from numpy import asarray
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost
import matplotlib.pyplot as plt
from WindPy import w
import pywt

#小波降噪函数
def wavelet_denoising(data, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    if threshold_type == 'hard':
        coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
    elif threshold_type == 'soft':
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

#滚动小波降噪函数
def rolling_wavelet_denoising(price_series, wavelet='db4', level=3, threshold_type='hard'):
    denoised_series = np.zeros_like(price_series)
    denoised_series[:-10] = wavelet_denoising(price_series[:-10], wavelet, level, threshold_type)
    for i in range(len(price_series) - 10, len(price_series)):
        denoised_series[i] = wavelet_denoising(price_series[:i + 1], wavelet, level, threshold_type)[-1]
    return denoised_series

#滚动因子筛选
def select_factors_with_xgboost(data, target_column, num_features_to_select,type):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if type == '分类':
        model = xgboost.XGBClassifier(objective='multi:softmax',num_class=3,enable_categorical=True)
    elif type == '回归':
        model = XGBRegressor(objective='reg:squarederror',enable_categorical=True)
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    selected_features = X.columns[np.argsort(feature_importances)[-num_features_to_select:]].tolist()
    return selected_features

#小波降噪试验集测试集划分
def train_test_split2(data, n_test):
    temp = data.iloc[:n_test+10,:].copy()
    for col in temp.columns[1:-1]:  
        price_series = temp[col].values
        # 进行滚动小波降噪
        denoised_price = rolling_wavelet_denoising(price_series, wavelet='db4', level=3, threshold_type='soft')
        temp[col] = denoised_price
    feature = temp.iloc[n_test-1000:n_test, :]  
    label = temp.iloc[n_test:n_test+10, :]      
    return feature, label

#滚动特征筛选测试集训练集划分
def train_test_split1(data, n_test,type):
    temp = data.iloc[:n_test+10,:].copy()
    selected_feature = select_factors_with_xgboost(temp.iloc[n_test-1000:n_test,1:],'label',20,type)
    temp = temp[['time_stamp']+selected_feature+['label']]
    train = temp.iloc[n_test-1000:n_test, :]
    test = temp.iloc[n_test:n_test+10, :]
    return train, test

#普通测试集训练集划分
def train_test_split3(data, n_test):
    temp = data.iloc[:n_test+10,:].copy()
    train = temp.iloc[n_test-1000:n_test, :]
    test = temp.iloc[n_test:n_test+10, :]
    return train, test

series = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\计算因子后.csv").drop(columns=['second','microsecond'])#此处drop的意义是，因为是10min级别，因此秒及毫秒已经无意义
factor_list_sigm = ['time_stamp',"benchPrice_sigm", "midPrice_sigm", "bl_dist_sigm", "b_avl_dist_sigm", "ml_dist_sigm", "m_avl_dist_sigm",
                    "ba_dist_sigm", "depth_sigm","tVol_depth_sigm", "tVol_abVol_sigm", "rskew_sigm", "downward_ratio_sigm", "reverse_sigm",
                    "pvol_corr_sigm",'AvgLastPrice_shift_ratio_sigm','CurrentTradeVol_sigm','CurrentTradeAmt_sigm','Return_sigm','l_avl_dist_sigm',
                    'unbalanced_vol','to_shortavg','to_longavg','year','month','day','hour','minute','AvgLastPrice','preclose','high','low','open','last','CurrentTradeVol',
                    'Bid_Price1','Bid_Amount1','Bid_Price2','Bid_Amount2','Offer_Price1','Offer_Amount1','Offer_Price2','Offer_Amount2','price_first_min','MA3','MA9','label']
factor_list = ['time_stamp',"bench_price", "midPrice", "bl_dist", "b_avl_dist", "ml_dist", "m_avl_dist",
                    "ba_dist", "depth","tVol_depth", "tVol_abVol", "rskew", "downward_ratio", "reverse",
                    "pvol_corr",'AvgLastPrice_shift_ratio','CurrentTradeAmt','Return','l_avl_dist',
                    'unbalanced_vol','to_shortavg','to_longavg','year','month','day','hour','minute','AvgLastPrice','preclose','high','low','open','last','CurrentTradeVol',
                    'Bid_Price1','Bid_Amount1','Bid_Price2','Bid_Amount2','Offer_Price1','Offer_Amount1','Offer_Price2','Offer_Amount2','price_first_min','MA3','MA9','label']
series = series[factor_list]
#cols_to_drop = ['time_stampshift','ret_negshift']  #此处是因为xgboost无法处理object类型的数据，会报错，删除
#series = series.drop(columns=cols_to_drop, errors='ignore')
series['time_stamp'] = pd.to_datetime(series['time_stamp'], format='%Y-%m-%d %H:%M:%S')
print(series['label'].value_counts(normalize=True))#统计各个标签的占比
data = series.copy()

#feature_importances：特征重要性字典；all_acc：整体准确率list；nz_acc：非零准确率list；nz：非零数量list；nz_pre:非零准确率list；nz_pre_num：非零数量list
feature_importances = {}
all_acc, nz_acc, nz , nz_pre, nz_pre_num= [], [], [], [], []
#此处创建回测的dataframe
backtest_data = pd.DataFrame(columns=['time_stamp', 'date', 'AvgLastPrice','last','CurrentTradeVol','Bid_Price1','Bid_Amount1','Bid_Price2','Bid_Amount2','Offer_Price1','Offer_Amount1','Offer_Price2','Offer_Amount2','price_first_min','MA3','MA9','label'])
for i in range(1010, 9500, 10):
    #对训练集和测试集进行划分
    train, test = train_test_split3(data, i)
    train1, test1 = train_test_split3(data, i)
    #训练模型
    model = XGBClassifier(objective='multi:softmax',num_class=3,enable_categorical=True)
    trainX, trainy = train.iloc[:, 1:-1], train.iloc[:, -1]
    model.fit(trainX, trainy)
    #准备回测的df
    temp = test1[['time_stamp', 'AvgLastPrice','last','CurrentTradeVol','Bid_Price1','Bid_Amount1','Bid_Price2','Bid_Amount2','Offer_Price1','Offer_Amount1','Offer_Price2','Offer_Amount2','price_first_min','MA3','MA9','label']]
    temp['date'] = temp['time_stamp'].dt.date
    temp = temp[['time_stamp', 'date','CurrentTradeVol', 'AvgLastPrice','last','Bid_Price1','Bid_Amount1','Bid_Price2','Bid_Amount2','Offer_Price1','Offer_Amount1','Offer_Price2','Offer_Amount2','price_first_min','MA3','MA9','label']]
    #pred：预测标签list；cn:预测正确数量；nzn：非零数量；nznc：非零且预测正确数量；precision：预测非零数量；precision1：预测非零且正确数量
    pred = []
    cn, nzn, nznc, precision, precision1 = 0, 0, 0, 0, 0
    #根据训练的模型进行滚动预测
    for j in range(len(test)):
        testx, testy = test.iloc[j, 1:-1], test.iloc[j, -1]
        yhat = model.predict(asarray([testx]))[0]
        pred.append(yhat)
        print('expected:', testy, 'predicted:',yhat)
        #如果预测值等于真实值，正确数量（cn）加一
        if testy == yhat:
            cn += 1
        #如果标签不等于1
        if testy != 1:
            nzn += 1
            if testy == yhat:
                nznc += 1
        #如果预测值不等于1
        if yhat != 1:
            precision += 1
            if testy == yhat:
                precision1 += 1
    #把每一期加入回测df
    temp['Prediction'] = pred
    backtest_data = pd.concat([backtest_data, temp], ignore_index=True)
    tc = cn/len(test)
    #计算每10期的准确率和非零准确率、非零精确率
    if nzn != 0:
        nc = nznc/nzn
    else:
        nc = 0
    if precision != 0:
        nc_p = precision1 / precision
    else:
        nc_p = 0
    print('整体准确率:', '%.3f' % tc, '非零准确率', '%.3f' % nc,'标签非零数量', nzn, '非零精确率', '%.3f' % nc_p, '预测非零数量',precision)
    print('已经进行到',i)
    all_acc.append(tc)
    nz_acc.append(nc)
    nz.append(nzn)
    nz_pre.append(nc_p)
    nz_pre_num.append(precision)
    #计算模型的特征重要性
    importance = model.feature_importances_
    for feature, imp in zip(train.columns[1:-1], importance):
        if feature in feature_importances:
            feature_importances[feature] += imp
        else:
            feature_importances[feature] = imp
#输出重要性排名前30的特征
top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:30]
print("Top 30 Features by Importance:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")

#将每次循环的数据保存为excel
record = pd.DataFrame({'总体准确率':all_acc, '非零准确率':nz_acc, '标签非零数量':nz, '非零精确率':nz_pre, '预测非零数量':nz_pre_num})
record['非零正确数量（准确率）'] = record['非零准确率'] * record['标签非零数量']
record['非零正确数量（精确率）'] = record['非零精确率'] * record['预测非零数量']
print("非零准确率:",record['非零正确数量（准确率）'].sum()/record['标签非零数量'].sum(), "非零精确率:", record['非零正确数量（精确率）'].sum()/record['预测非零数量'].sum(), "整体准确率:", record['总体准确率'].mean())
record.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\循环结果.csv")
backtest_data['date'] = backtest_data['date'].astype(str)
backtest_data.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\回测数据.csv")

#%%回测模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

backtest_data = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\回测数据.csv")
backtest_data['time_stamp'] = pd.to_datetime(backtest_data['time_stamp'])
backtest_data['date'] = pd.to_datetime(backtest_data['date'])

#假设本金为1000000，并计算初始拥有的数量
j = 1000000
initial_amount = j/backtest_data.loc[0, 'Offer_Price1']

#对均线进行二次计算均值
backtest_data['MA3'] = backtest_data['MA3'].rolling(window=6,min_periods=2).mean()
backtest_data['MA9'] = backtest_data['MA9'].rolling(window=30,min_periods=2).mean()

#计算一个list中最后n个数的均值
def mean_of_last_n(nums, n):
    if not nums or n <= 0:
        return None  
    last_n = nums[-n:]
    return sum(last_n) / len(last_n)

#找到最大的局部最大值，也就是大于左右两侧值的值
def find_local_maxima(lst):

    local_max = []
    n = len(lst)
    
    if n < 2:
        return local_max
    
    for i in range(n):
        if i == 0:
            if lst[i] > lst[i+1]:
                local_max.append(lst[i])
        elif i == n-1:
            if lst[i] > lst[i-1]:
                local_max.append(lst[i])
        else:
            if lst[i] > lst[i-1] and lst[i] > lst[i+1]:
                local_max.append(lst[i])
    return max(local_max)

#合约切换日期，在该日期，基准和组合都需要切换合约
switch_date_list = [datetime(2024, 2, 15),
    datetime(2024, 5, 15),
    datetime(2024, 8, 15),
    datetime(2024, 11, 15),
    datetime(2025, 2, 15),
    datetime(2025, 5, 15),]

# 初始化回测参数
initial_cash = 1000000 #初始现金
cash = initial_cash
position = 0           #持仓数量
portfolio_value = []   #组合净值list
base_value = []        #基准净值list
cashlist = []          #现金list
positionlist = []      #持仓list
returnlist = []        #组合收益率list
labellist = []         #真实标签list
predlist = []          #预测标签list
holdinglist = []       #持仓list（代表持仓于本金几倍，例如1代表满仓，0.5代表半仓）
pricelist = []         #均价list
alphalist = []         #超额收益list
buylist = []           #1代表有买入操作，0为没有买入操作
selllist = []          #1代表有卖出操作，0为没有卖出操作
base_vol = []          #基准持仓数量
dealing_frequency = 0  #交易次数
holding = 0            #持仓
count = 0              #后面设置了止损模块，设置了最多每150个切片止损一次，count用来计时
base = 0               #基准净值
backtest_data['price_first_min'] = backtest_data['price_first_min'].shift(-1) #第一分钟的价格需要shift一下来进行操作
probability = 0.63               #初始化准确率阈值，如果最近30期的准确率超过该值则认为模型最近准确率较高
excess_return_threshold = -0.5   #若超额收益减少超过50%，则平仓
action = 0                       #若有买入为1，卖出为-1，无行动为0

# 回测逻辑
for idx, row in backtest_data.iterrows():
    #定义每一期的各个特征
    price = row['AvgLastPrice']
    last_price = row['last']
    date = row['date']
    MA_3 = row['MA3']
    MA_9 = row['MA9']
    first1_price = row['price_first_min']/10000
    amt = row['CurrentTradeVol']
    bid_price = row['Bid_Price1']
    bid_amount = row['Bid_Amount1']
    bid_price2 = row['Bid_Price2']
    bid_amount2 = row['Bid_Amount2']
    offer_price = row['Offer_Price1']
    offer_amount = row['Offer_Amount1']
    offer_price2 = row['Offer_Price2']
    offer_amount2 = row['Offer_Amount2']
    true = row['label']
    signal = row['Prediction']
    
    #如果到了变更合约的日期，则按照上一期的价格卖出并按照改期价格买入
    if date in switch_date_list and backtest_data.loc[idx-1,'date'] != date:
        initial_amount = base / price

        temp = position
        cash += position * backtest_data.loc[idx-1,'AvgLastPrice']
        cash -= position * price
    
    #计算组合及基准的净值
    base = initial_amount * price
    current_value = cash + position * price
    alpha = current_value - base
    
    #计算最近30期的非零准确率
    if idx <= 29:
        labellist.append(true)
        predlist.append(signal)
    else:
        templist1 = labellist[-30:]
        templist2 = predlist[-30:]
        count_not_2 = sum(1 for x in templist1 if x != 1)
        count_equal = sum(1 for i in range(len(templist1)) if templist1[i] != 1 and templist1[i] == templist2[i])
        probability = count_equal / count_not_2
        labellist.append(true)
        predlist.append(signal)
    
    #如果短均线大于长均线，且短均线边际增长时，若模型给出预测将继续上升，且模型近期预测表现不错，则买入
    if (idx == 0) or((MA_3>=MA_9) and (MA_3>=backtest_data.loc[idx-1,'MA3'])):
        if signal == 2   and  probability > 0.33 :
            if holding < 1 and idx + 1 < len(backtest_data):
                position +=  0.5*initial_cash / backtest_data.loc[idx+1,'AvgLastPrice']
                cash -=  0.5*initial_cash
                holding += 0.5
                dealing_frequency += 1
                action = 1
    
    #如果短均线小于长均线，且短均线边际下降时，若模型给出预测将继续下降，且模型近期预测表现不错，则卖出
    elif (idx == 0) or((MA_3<MA_9) and (MA_3<backtest_data.loc[idx-1,'MA3'])):
        if signal == 0   and  probability > 0.33:
            if holding > -1 and idx + 1 < len(backtest_data):
                position -= 0.5*initial_cash / backtest_data.loc[idx+1,'AvgLastPrice']
                cash += 0.5*initial_cash
                holding -= 0.5
                dealing_frequency += 1
                action = -1
    
    #将一些相关数据加入list中
    portfolio_value.append(current_value)
    base_value.append(base)
    returnlist.append(current_value)
    positionlist.append(position)
    cashlist.append(cash)
    holdinglist.append(holding)
    labellist.append(true)
    alphalist.append(alpha)
    base_vol.append(initial_amount)
    count += 1

    if action == 1:
        buylist.append(1)
        selllist.append(0)
    elif action == -1:
        buylist.append(0)
        selllist.append(1)
    else:
        buylist.append(0)
        selllist.append(0)
    action = 0



# 计算回测指标，并将一些list变成df中的列方便保存后查看
backtest_data['portfolio_value'] = portfolio_value
backtest_data['base_value'] = base_value
backtest_data['return'] = returnlist
backtest_data['cash'] = cashlist
backtest_data['position'] = positionlist
backtest_data['holding'] = holdinglist
backtest_data['alpha'] = alphalist
backtest_data['buy'] = buylist
backtest_data['sell'] = selllist
backtest_data['base_amt'] = base_vol

#计算总收益、年化收益、夏普比率，最大回撤（1.52%当作无风险利率）
total_return = (backtest_data['portfolio_value'].iloc[-1] - backtest_data['portfolio_value'].iloc[0] ) / backtest_data['portfolio_value'].iloc[0]
base_return = (backtest_data['base_value'].iloc[-1] - backtest_data['base_value'].iloc[0] ) / backtest_data['base_value'].iloc[0]
annualized_return = total_return * (252 * 25 / len(backtest_data)) 
annualized_std = backtest_data['portfolio_value'].pct_change().std() * np.sqrt(252 * 25)
sharpe_ratio = (annualized_return - 0.0152) / annualized_std
annualized_return = base_return * (252 * 25 / len(backtest_data))
annualized_std = backtest_data['base_value'].pct_change().std() * np.sqrt(252 * 25)
sharpe_ratio1 = (annualized_return - 0.0152) / annualized_std
print('交易次数:',dealing_frequency)
max_drawdown = (backtest_data['portfolio_value'].cummax() - backtest_data['portfolio_value']).max() / backtest_data['portfolio_value'].cummax().max()
max_drawdown1 = (backtest_data['base_value'].cummax() - backtest_data['base_value']).max() / backtest_data['base_value'].cummax().max()
backtest_data.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\回测结果.csv")


# 输出结果
print(f"Total Return: {total_return:.2%}")
print(f"Base Return: {base_return:.2%}")
print(f"Our_Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Base_Sharpe Ratio: {sharpe_ratio1:.2f}")
print(f"Our_Max_Drawdown: {max_drawdown:.2%}")
print(f"Base_Max_Drawdown: {max_drawdown1:.2%}")

# 绘制资产净值曲线和超额收益时序图
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))
plt.plot(backtest_data['time_stamp'],backtest_data['portfolio_value'], label='策略收益')
plt.plot(backtest_data['time_stamp'],backtest_data['base_value'], label='基准收益')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.title('回测收益时序图')
plt.legend()
plt.savefig(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\回测收益.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(backtest_data['time_stamp'],backtest_data['alpha'], label='超额收益')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.title('超额收益历史时序图')
plt.legend()
plt.savefig(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\超额收益.png")
plt.show()
plt.close()
