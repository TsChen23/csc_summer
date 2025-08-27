import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

backtest_data = pd.read_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Input\计算因子后.csv").iloc[1010:,].reset_index(drop=True)
backtest_data['position_diff'] = backtest_data['total_position'].diff()
backtest_data['time_stamp'] = pd.to_datetime(backtest_data['time_stamp'])
j = 1000000
initial_amount = j/backtest_data.loc[0, 'Offer_Price1']
backtest_data['base_value'] = backtest_data['AvgLastPrice'] * initial_amount

def mean_of_last_n(nums, n):
    if not nums or n <= 0:
        return None  
    last_n = nums[-n:]
    return sum(last_n) / len(last_n)


# 初始化回测参数
initial_cash = 1000000
flag = 0
cash = initial_cash
position = 0
portfolio_value = []  
cashlist = []
positionlist = []
returnlist = []
labellist = []
holdinglist = []
pricelist = []
alphalist = []
thershold = 0.03
dealing_frequency = 0
holding = 0
count = 0
backtest_data['price_first_min'] = backtest_data['price_first_min'].shift(-1)
lastprice = 0
probability = 0.63
excess_return_threshold = -0.40
# 回测逻辑

for idx, row in backtest_data.iterrows():
    price = row['AvgLastPrice']
    last_price = row['last']
    position_diff = row['position_diff']
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
    base = row['base_value']

    current_value = cash + position * price
    alpha = current_value - base
    
    if idx == 0:
        portfolio_value.append(current_value)
        returnlist.append(current_value)
        positionlist.append(position)
        cashlist.append(cash)
        holdinglist.append(holding)
        labellist.append(true)
        alphalist.append(alpha)
        count += 1
        continue

    if (idx == 0) or((MA_3>=MA_9) and (MA_3>=backtest_data.loc[idx-1,'MA3'])):
            if holding < 1:
                position += 0.5*initial_cash / backtest_data.loc[idx+1,'AvgLastPrice']
                cash -= 0.5*initial_cash
                holding += 0.5
                dealing_frequency += 1
                action = 1

    elif (idx == 0) or((MA_3<MA_9) and (MA_3<backtest_data.loc[idx-1,'MA3'])):
            if holding > -1:
                position -= 0.5*initial_cash / backtest_data.loc[idx+1,'AvgLastPrice']
                cash += 0.5*initial_cash
                holding -= 0.5
                dealing_frequency += 1
                action = -1
    
    
    portfolio_value.append(current_value)
    returnlist.append(current_value)
    positionlist.append(position)
    cashlist.append(cash)
    holdinglist.append(holding)
    labellist.append(true)
    alphalist.append(alpha)
    count += 1


# 计算回测指标
backtest_data['portfolio_value'] = portfolio_value
backtest_data['return'] = returnlist
backtest_data['cash'] = cashlist
backtest_data['position'] = positionlist
backtest_data['holding'] = holdinglist
backtest_data['alpha'] = alphalist
total_return = (backtest_data['portfolio_value'].iloc[-1] - backtest_data['portfolio_value'].iloc[0] ) / backtest_data['portfolio_value'].iloc[0]
base_return = (backtest_data['base_value'].iloc[-1] - backtest_data['base_value'].iloc[0] ) / backtest_data['base_value'].iloc[0]
annualized_return = total_return * (252 * 13 / len(backtest_data)) 
annualized_std = backtest_data['portfolio_value'].pct_change().std() * np.sqrt(252 * 13)
sharpe_ratio = (annualized_return - 0.0152) / annualized_std
annualized_return = base_return * (252 * 13 / len(backtest_data))
annualized_std = backtest_data['base_value'].pct_change().std() * np.sqrt(252 * 13)
sharpe_ratio1 = (annualized_return - 0.0152) / annualized_std
chaoe = 0.013 * (252 * 13 / len(backtest_data)) 
annualized_std = backtest_data['alpha'].std() * np.sqrt(252 * 13)
sharpe_ratio2 = chaoe  / annualized_std
print('交易次数:',dealing_frequency)
max_drawdown = (backtest_data['portfolio_value'].cummax() - backtest_data['portfolio_value']).max() / backtest_data['portfolio_value'].cummax().max()
max_drawdown1 = (backtest_data['base_value'].cummax() - backtest_data['base_value']).max() / backtest_data['base_value'].cummax().max()
backtest_data.to_csv(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\不带模型版回测结果.csv")


# 输出结果
print(f"Total Return: {total_return:.2%}")
print(f"Base Return: {base_return:.2%}")
print(f"Our_Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Base_Sharpe Ratio: {sharpe_ratio1:.2f}")
print(f"chaoe_Sharpe Ratio: {sharpe_ratio2:.2f}")
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
plt.savefig(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\回测收益（不带模型）.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(backtest_data['time_stamp'],backtest_data['alpha'], label='超额收益')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.title('超额收益历史时序图')
plt.legend()
plt.savefig(r"C:\Users\chens\Desktop\中信建投暑期\国债期货量化课题\Output\结果\超额收益（不带模型）.png")
plt.show()
plt.close()
