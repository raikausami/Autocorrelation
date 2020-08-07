import pandas as pd
import numpy as np
import math
import  matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('1.csv', index_col='date',parse_dates=True,dtype='float')
ts = data['num']
ts.head()
data_length=len(ts)

ts_acf = sm.tsa.stattools.acf(ts, nlags=40)


#method要検討
ts_pacf = sm.tsa.stattools.pacf(ts,nlags=7,method='ols',alpha = None)

#棄却域
print(1.96/math.sqrt(data_length))
rejection_area = 1.96/math.sqrt(data_length)

for i  in range(len(ts_pacf)):
    if i>0 and ts_pacf[i] > rejection_area:
        print(i,ts_pacf[i])

sm.graphics.tsa.plot_pacf(ts)

plt.show()
