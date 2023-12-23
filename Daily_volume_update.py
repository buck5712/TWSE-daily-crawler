from datetime import date
from datetime import timedelta
import pandas as pd
import numpy as np

def get_date():
    # Get today's date
    today = date.today()
    
    # Last date
    if today.weekday()==0:
        last_day = today - timedelta(days = 3)
    else:
        last_day = today - timedelta(days = 1)  

    return str(today).replace('-',''), str(last_day).replace('-','')


today, _ = get_date()
stock_list=['1101', '1216', '1301', '1303', '2303', '2308', '2330', '2382', '2408','2412', '2454', '2615', '2801', '2880', '2881', '2882', '2884', '2885', '2887', '2892', '3034', '5880', '3711', '2002', '2886']

for stock in stock_list:
    selected=pd.read_csv(f'D:/交大/Paper/stock price/Data/籌碼資料/2020_2023/selected/{stock}_broker_volume_repeat.csv', index_col='日期')
    selected.index=pd.to_datetime(selected.index)

    vol=pd.read_csv(f'D:\\VolumeCrawler\\{today}\\{stock}.csv', names=list(np.arange(1,12)), encoding='big5')
    vol.drop([0,1], axis=0, inplace=True)
    vol.drop(6, axis=1, inplace=True)
    left=vol.iloc[:,0:5].reset_index(drop=True)
    right=vol.iloc[:,5:].reset_index(drop=True)
    left.columns=left.iloc[0,:]
    left.drop(0, inplace=True)
    right.columns=right.iloc[0,:]
    right.drop(0, inplace=True)
    right.dropna(inplace=True)
    vol=pd.concat([left, right], axis=0)

    new = pd.DataFrame(0, index=[pd.to_datetime(today)], columns=selected.columns)

    for i in range(len(vol)):
        if vol.iloc[i,1][:4] in selected.columns:
            new.loc[today, vol.iloc[i,1][:4]]+=(int(vol.iloc[i,3])-int(vol.iloc[i,4]))/1000
    selected=pd.concat([selected, new])
    selected.index.name=('日期')
    selected.to_csv(f'D:/交大/Paper/stock price/Data/籌碼資料/2020_2023/selected/{stock}_broker_volume_repeat.csv', index=True)
# while True:
