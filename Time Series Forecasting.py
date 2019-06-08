
#----Import package------------------------------------------------------------#
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import Holt
import statsmodels.tsa.api as sm
#----Read the input file-------------------------------------------------------#

def parser(x):
	return datetime.strptime(x, '%Y-%m')

RequiredDF = pd.read_csv('AirPassengers.csv', parse_dates=[0], 
                         index_col='Month',date_parser=parser)

TS=RequiredDF["#Passengers"]
TS.head
TS['1949-01-01':'1949-05-01']
TS['1949']

#---Check stationary of series-------------------------------------------------#
#---Mean,variance and covariance should not be depend  of time-----------------#
#A TS is said to be stationary if its statistical properties such as mean, variance remain constant over time
plt.plot(TS)
#Perform Dickey-Fuller test:
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(TS)


#----------- Making time series as stationary----------------------------------#

TS_log=np.log(TS)
plt.plot(TS_log)
#---------------Smooting Moving Average----------------------------------------#
moving_avg = TS_log.rolling(window=12,center=False).mean()
plt.plot(moving_avg)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = TS_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

#----------------Exponential smooting average----------------------------------#
expwighted_avg = pd.ewma(TS_log, halflife=12)
plt.plot(TS_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = TS_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)



#----Simple trend technique that we discussed before sometime don't work-------#

ts_log_diff = TS_log - TS_log.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

decomposition = seasonal_decompose(TS_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(TS_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)







plot_acf(ts_log_diff,lags=20)
plt.show()    # from plot , q should 1

plot_pacf(ts_log_diff, lags=20)
plt.show()   # from plot , p should 2

model = ARIMA(TS_log, order=(2, 1, 2))  
results_ARIMA = model.fit() 
results_ARIMA.summary()


predictions = results_ARIMA.predict('1949-02-01', '1960-12-01',typ='levels')
# Compute the mean square error
mse = ((predictions -TS_log ) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

Forcasted_value= pd.Series(np.exp(results_ARIMA.forecast(steps=50, exog=None, alpha=0.05)[0]))

plt.plot(TS)
plt.plot(Forcasted_value)
plt.show()
