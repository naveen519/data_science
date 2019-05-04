### read the data from sources 

kings = scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
births = scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")
souvenir = scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
rain = scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
skirts = scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5)

### Forecasting 
kings
library(forecast)
library(TTR)
library(fpp2)
library(ggplot2)
### install forecast and TTR packages 
head(kings)
kingstimeseries = ts(kings)
kingstimeseries
autoplot(kingstimeseries)
## Timeseries plot of kings timeseries 
plot.ts(kingstimeseries)
## Births timeseries 
births

## convert birth data to time series, starting from Jan, 1946
head(births)
birthstimeseries = ts(births, frequency=12, start=c(1946,1))
birthstimeseries

## Timeseries plot of birth series, indicates seasionality 
plot.ts(birthstimeseries)

### decompose birth timeseries 
birthdecomp = decompose(birthstimeseries, type = "additive")
birthdemop = decompose(birthstimeseries, type = "multiplicative")
### plot decomposition of birth timeseries 

autoplot(birthdecomp)
autoplot(birthdemop)

### souvenier time series( Non stationary)
souvenir

souvenirtimeseries = ts(souvenir, frequency=12, start=c(1987,1))
souvenirtimeseries
## Non stationary series with spikes increasing 
plot.ts(souvenirtimeseries)

sovseiresdecom = decompose(souvenirtimeseries, type="multiplicative")

plot(sovseiresdecom) 

logsovseries = log(souvenirtimeseries)
plot.ts(logsovseries)
## Log conversion to bring the spikes magnitude down 
logsouvenirtimeseries = log(souvenirtimeseries)
plot.ts(logsouvenirtimeseries)

### Simple moving averages 

plot.ts(kingstimeseries)
library("TTR")
library("forecast")

## simple moving averages of order 3 
kingstimeseriesSMA3 = SMA(kingstimeseries,n=3)
kingstimeseriesSMA3
plot.ts(kingstimeseriesSMA3)
plot.ts(kingstimeseries)
kingstimeseries
kingstimeseriesSMA5 = SMA(kingstimeseries,n=5)
plot.ts(kingstimeseriesSMA5)

kingstimeseriesSMA11 = SMA(kingstimeseries,n=11)
plot.ts(kingstimeseriesSMA11)


## compare the origina kings timeseries with Moving averages 

kingstimeseries

kingstimeseriesSMA3

kingstimeseriesSMA3  = SMA(kingstimeseries,n=3)
plot.ts(kingstimeseriesSMA3)

#### holtwinters on kings series 
?HoltWinters

kingsholtwinters = HoltWinters(kingstimeseries, beta = F, gamma=F)
?HoltWinters

plot(kingsholtwinters)

kingsholtwinters

forecasts = forecast:::forecast.HoltWinters(kingsholtwinters, h = 12)
autoplot(forecasts)

forecasts
## Decomposing time series 
plot.ts(birthstimeseries)

birthstimeseriescomponents = decompose(birthstimeseries)
birthdecom = decompose(birthstimeseries)
plot(birthdecom)
plot(birthstimeseriescomponents)
birthstimeseriescomponents

### seasonality 
birthstimeseriescomponents$seasonal

## trend 

birthstimeseriescomponents$trend

## seasionality adjusting ( remove sesionality to check the trend and randomness)
birthstimeseriescomponents = decompose(birthstimeseries)
birthstimeseriesseasonallyadjusted = birthstimeseries - birthstimeseriescomponents$seasonal

plot(birthstimeseriesseasonallyadjusted) ## seasionality adjusted 

### 

#### holtwinters exponential smoothing on birthseries 
plot.ts(birthstimeseries)
birthhw = HoltWinters(birthstimeseries ,beta = F,  gamma = F)

plot(birthhw)

birthshwforecasts = forecast:::forecast.HoltWinters(birthhw, h=12)
autoplot(birthshwforecasts)

## create a training and test dataset 

plot.ts(birthstimeseries)

birthtrain = window(birthstimeseries, start =c(1946,1), end=c(1958,12))
birthtest = window(birthstimeseries, start=c(1959,1))

birthhw = HoltWinters(birthtrain, beta=F, gamma=F) ## exp smoothing

birthfor = forecast(birthhw, h=12)

### compute error on the test period 

err = birthtest - birthfor$mean

rmse = sqrt(mean(err**2))
rmse

### Mape 

mape = mean((abs(err)/birthtest)*100)

mape
## include beta 

birthhw_beta = HoltWinters(birthtrain,  gamma=F)

birthfor = forecast(birthhw_beta, h=12)


### compute error on the test period 

err = birthtest - birthfor$mean

rmse = sqrt(mean(err**2))
rmse

### Mape 

mape = mean((abs(err)/birthtest)*100)

mape

### forecast using hw model + beta + seasionality

birthsforecast = HoltWinters(birthtrain)
plot(birthsforecast)
birthfor = forecast(birthsforecast, h=12)
autoplot(birthfor)
### compute error on the test period 

err = birthtest - birthfor$mean

rmse = sqrt(mean(err**2))
rmse

### Mape 

mape = mean((abs(err)/birthtest)*100)

mape

### HW forecasts with multipilicative 

birthhw_mul = HoltWinters(birthtrain, seasonal = "multiplicative")
plot(birthhw_mul)
birthfor = forecast(birthhw_mul, h=12)
autoplot(birthfor)
### compute error on the test period 

err = birthtest - birthfor$mean

rmse = sqrt(mean(err**2))
rmse

### Mape 

mape = mean((abs(err)/birthtest)*100)

mape

### additive model on Sov timeseries 

sovtrain = window(souvenirtimeseries, end = c(1992,12))
sovtest = window(souvenirtimeseries, start = c(1993,1))

## additive model 

sov_hw_add = HoltWinters(sovtrain, seasonal = "additive")
plot(sov_hw_add)
sovfor = forecast(sov_hw_add, h=12)

autoplot(sovfor)

### RMSE 

err = sovtest - sovfor$mean

rmse = sqrt(mean(err**2))
rmse

mape = mean((abs(err)/sovtest)*100)
mape

## performance of multiplicative model 

sov_hw_mul = HoltWinters(sovtrain, seasonal = "multiplicative")
plot(sov_hw_mul)
sovfor = forecast(sov_hw_mul, h=12)
autoplot(sovfor)
### RMSE 

err = sovtest - sovfor$mean

rmse = sqrt(mean(err**2))
rmse

mape = mean((abs(err)/sovtest)*100)
mape

### Rainfall data 
rainfall_london = ts(rain, start = 1901)

plot(rainfall_london)

rainfall_decom = decompose(rainfall_london)

raindiff1 = diff(rainfall_london, 1)

plot(raindiff1)

raindiff2 = diff(rainfall_london, 2)

plot(raindiff2)
## forecast using 

rainfall_lon_hw = HoltWinters(rainfall_london, gamma=F)

#### forecast rainfall using Holtwinters 

rain_fore = forecast.HoltWinters(rainfall_lon_hw, h=10)

plot.forecast(rain_fore)


### Arima on Sk sreis

skseries = ts(skirts)
plot(skseries)

###  non stationary timeseries - differentiating 

skseriesdiff1 = diff(skseries, 1)

plot.ts(skseriesdiff1)

skseriesdiff2 = diff(skseries, 2)

plot.ts(skseriesdiff2)

## adf test to check stationarity of a time series 

library(tseries)
?adf.test

adf.test(skseries, alternative = "stationary", k=0)
adf.test(skseriesdiff1, alternative = "stationary", k=0)
adf.test(skseriesdiff2, alternative = "stationary", k=0)

## sov timeseries 

plot.ts(souvenirtimeseries)

adf.test(souvenirtimeseries, alternative = "stationary")

sovdiff1 = diff(souvenirtimeseries, 1)

adf.test(sovdiff1, alternative = "stationary" )

plot.ts(sovdiff1)

### values of p,d q

## auto.arima 
 
sov_arima = auto.arima(sovtrain, seasonal = F)

### forecast for year 1993 

sovfor = forecast(sov_arima, h=12)

err = sovtest - sovfor$mean

mape = mean((abs(err)/sovtest)*100)
mape

### auto.arima with seasonality 

sov.arima = auto.arima(sovtrain, seasonal = F)

sovfor = forecast(sov.arima, h=12)

## performance 

err = sovtest - sovfor$mean

mape = mean((abs(err)/sovtest)*100)
mape 

### Values of p and q 

acf(sovtrain)
pacf(sovtrain)

## arima(p,d,q)

sov_arima = arima(sovtrain, c(1,1,1), seasonal = c(1,1,1))

sovfor = forecast(sov_arima, h=12)

## err and  mape 

err = sovtest - sovfor$mean

mape = mean((abs(err)/sovtest)*100)

mape

sov_arima = arima(sovtrain, c(1,1,1), seasonal = c(0,1,1))

sovfor = forecast(sov_arima, h=12)

err = sovtest - sovfor$mean

mape = mean((abs(err)/sovtest)*100)

mape

#### rainseries 

plot(rain)

## Holtwinters on sk data

sk.hw = HoltWinters(skseries, gamma=F)

 sk.foreca = forecast.HoltWinters(sk.hw, h = 6)

 plot.forecast(sk.foreca)
 
 ### ARIMA(p,d,q)
 
 plot.ts(skseries)
 
 
 sk.diff1 = diff(skseries, lag=1)
 
 plot.ts(sk.diff1)
 
 sk.diff2 = diff(skseries, lag=2)

plot.ts(sk.diff2)

### Auto.arima 

sk.arima = auto.arima(skseries)

sk.arima

sk.ar.fore = forecast:::forecast.Arima(sk.arima,  h = 6)

forecast:::plot.forecast(sk.ar.fore)

 ### first order differences 

skirtsdiff1 = diff(skirtsseries, 1) ## 1 for first order differences 


plot(skirtsdiff1)

skirtsdiff2 = diff(skirtsseries, 2)

plot(skirtsdiff2)

##sovenier timeseries differences 

sov_diff1 = diff(souvenirtimeseries, 1)

plot(sov_diff1)

Sov_diff2 = diff(souvenirtimeseries, 2)

plot(Sov_diff2)


 sov.arima = auto.arima(souvenirtimeseries)
 
 asov.ar.forec = forecast.Arima(sov.arima, h=12)
 
 plot.forecast(asov.ar.forec)

### p and q 

pacf(souvenirtimeseries)

acf(souvenirtimeseries)

### Souv timeseries excluding the seasionality 

sov.arima.noseas = auto.arima(souvenirtimeseries, seasonal = F)

sov.arima.noseas.fore = forecast.Arima(sov.arima.noseas)

plot.forecast(sov.arima.noseas.fore)

?auto.arima

## rainseries 

## auto.arima 

rainforec = HoltWinters(rainseries, beta=F, gamma = F)

plot(rainforec)

rainarima = auto.arima(rainseries)

rainarima

#### arima(p,d,q)

rainarima = arima(rainseries, c(1,0,1))

### Chennai rainfall ARIMA prediction 

setwd("D:/AP/Forecasting")

rainfall = read.csv("Chennai_rainfall.csv", sep=",")

head(rainfall)


rain = rainfall$Rain

chennirain = ts(rain, frequency = 12, start = c(2000,1))

plot.ts(chennirain)

### Decomposint the rain series 

raindecom = decompose(chennirain)

plot(raindecom)

### holtwinters model 

rainfalltrain = window(chennirain, end = c(2016,4))
rainfalltest = window(chennirain, start = c(2016,5), end=c(2017,4))

### holtwinters model 

rainfallmodel = HoltWinters(rainfalltrain)

plot(rainfallmodel)

rainfalltrain[ rainfalltrain < 1] = 2

rainfallmodel_mul = HoltWinters(rainfalltrain, seasonal = "multiplicative")

plot(rainfallmodel_mul)


rainfalltest[ rainfalltest < 1] = 2

### forecast for the next 12 months 

rainforec = forecast(rainfallmodel_mul, h=12)

## err 

err = rainfalltest - rainforec$mean

mape = mean((abs(err)/rainfalltest)*100)
mape
 
### acf 

acf(chennirain)
pacf(chennirain)

## stationarity check 

adf.test(chennirain, alternative="stationary")

rainfalldiff1 = diff(chennirain, 1)

plot(rainfalldiff1)

adf.test(rainfalldiff1, alternative = "stationary")

d=1 

acf(rainfalldiff1)
pacf(rainfalldiff1)


### auto.arima on chennai rainfall data 

rain.arima = auto.arima(rainfalltrain)

### forecasts for next 12 months 

rain_forec = forecast( rain.arima, h = 12)

rainforec = forecast(rain_forec, h=12)

## Model performance 

err = rainfalltest - rainforec$mean

mape = mean((abs(err)/rainfalltest)*100)



### Arimax model on Chennai Ranfall data

require(gridExtra)
library(ggplot2)
p1 = ggplot(rainfall, aes(x = X2, y = Rain)) +
  ylab("Rainfall") +
  xlab("Years") +
  geom_line(color = "Red") +
  coord_cartesian( xlim = c(2000,2012))+
  expand_limits(x = 0, y = 0)+
  ggtitle("IOD Vs rainfall")
p1


p2 = ggplot(rainfall, aes(x = X2, y = IOD)) +
  ylab("IOD") +
  xlab("Year") +
  coord_cartesian( xlim = c(2000,2012))+
  geom_line(color = "blue" ) +
  expand_limits(x = 0, y = 0)

grid.arrange(p1, p2, ncol=1, nrow=2)

cor(rainfall$Rain, rainfall$IOD)

### subset rainfall data for first 137 records( IOD is present)

rainfall2 = rainfall[1:137, ]

cor(rainfall2$Rain, rainfall2$IOD)

cor(rainfall2$Rain, rainfall2$QBO)

####  Arimax model 

rainseries2 = ts(rainfall2$Rain, frequency = 12, start=c(2000,1))

plot.ts(rainseries2)

# Auto.arima to pick values of p,d,q + Seasonality 

ran.arima2 = auto.arima(rainseries2)

ran.arima2

rain.arima2.forec = forecast(ran.arima2, h = 12)

rain2test = window(chennirain, start = c(2011, 6), end = c(2012,5))

### forecast error 

rain2test[rain2test < 1] = 2

err = rain2test - rain.arima2.forec$mean 

## mape 

mape = mean((abs(err)/rain2test)*100)

## Arimax with input variables 

IOD = rainfall2$IOD

rainfall.arimax = auto.arima(rainseries2, xreg = IOD )

future_IOD = c(0.261, 0.533, 0.646, 0.586, 0.735, 0.597,0.097,0.227, 0.1, 0.195, -0.065, -0.124)

arimax.forecasts = forecast(rainfall.arimax, h = 12, xreg = c(future_IOD))

### measure the erro with IOD values 

err = rain2test - arimax.forecasts$mean

mape = mean((abs(err)/rain2test)*100)


### include QBO values into the xreg  

QBO = rainfall2$QBO

IOD_QBO = rainfall2[, 6:7]

rainfall.arimax2 = auto.arima(rainseries2, xreg=IOD_QBO)

#### Subset IOD and QBO values 

IOD_QBO = rainfall2[, 6:7]

rainfall.arimax2 = auto.arima(rainseries2, xreg=IOD_QBO)
#### select IOD and QBO values for Dec2010 to May2011 
IOD_QBO_fut = rainfall[132:137, 6:7]
rain.forecats2 = forecast.Arima(rainfall.arimax2, h = 6,xreg = IOD_QBO_fut )

rain.forecats2

