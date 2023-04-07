import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matrixprofile as mp
df=pd.read_csv("F:/python_code/data/article-matrix-profile-intro/data/nyc_yellow_taxi_passenger_count_2018_hourly.csv")
df['pickup_name'] = pd.to_datetime(df['pickup_datetime'])#将所得到的数据转换成时间帧 
df = df.set_index('pickup_datetime').sort_index()
df.head()
df['passenger_count'].describe()

df.plot(figsize=(20,7),legend=None,title='2018 NYC Hourly Passenger Count\nYellow Taxi Cab Only')
plt.xlabel('Pickup Datetimee')
plt.ylabel('Passengers')
plt.show()


#compute mp
windows=[('4 Hours', 4),
    ('8 Hours', 8),
    ('12 Hours', 12),
    ('24 Hours', 24),
    ('7 Days', 7 * 24),
    ('30 Days', 30 * 24),]
profiles={}

for label,window_size in windows:
    profile = mp.compute(df['passenger_count'].values,window_size)
    key='{} Profile'.format(label)
    profiles[key]=profile

#将plot matrix profile

fig,axes=plt.subplots(6,1,sharex=True,figsize=(15,10))

for ax_index,window in enumerate(windows):
    key='{} Profile'.format(window[0])
    profile=profiles[key]
    axes[ax_index].plot(profile['mp'])
    axes[ax_index].set_title(key)
plt.xlabel('Pickup Datetime')
plt.tight_layout()
plt.show



#find Discords
for label, window_size in windows:
    key = '{} Profile'.format(label)
    profiles[key] = mp.discover.discords(profiles[key], k=5)
    
    window_size = profiles[key]['w']
    mp_adjusted = np.append(profiles[key]['mp'], np.zeros(window_size - 1) + np.nan)
    
    plt.figure(figsize=(15, 7))
    ax = plt.plot(df.index.values, mp_adjusted)
    plt.title(key)
    
    for start_index in profiles[key]['discords']:
        x = df.index.values[start_index:start_index+window_size]
        y = mp_adjusted[start_index:start_index+window_size]
        plt.plot(x, y, c='r')
    
    plt.show()