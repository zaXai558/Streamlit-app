import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
#import plotly.graph_objs as go 

# using to write in app # makes text bigger and ## makes text smaller
st.write("# Your Data")
#load data         
df =pd.read_excel('HandsOnTrial_Mayukh.xlsx')
#make a copy
train=df.copy()
#rename some variables
train.rename(columns = {'PO Date':'DATE'}, inplace = True)
train.rename(columns = {'Invoice Value (D)':'Invoice'}, inplace = True)
#display data in app 
if st.sidebar.checkbox('Show Raw Data'):
    '## Raw Data',train
#Aggregate the data to just Invoice 
train=train.groupby(train['DATE'].dt.date).sum() 
data = {}
data["DATE"] = pd.date_range("2016-04-01", "2016-06-30", freq="D")

finaldf = pd.DataFrame(data=data)
finaldf = finaldf.set_index("DATE")
finaldf = finaldf.merge(train, left_index=True, right_index=True, how="left")
finaldf=finaldf.dropna()
unidf = finaldf.drop(columns=['Quantity (A)','Quantity','Unit Rate','To be del. (QTY)\nB','To be del. (Value)','To be inv. (QTY)','To be inv. (C )'])
   
if st.sidebar.checkbox('Timeseries data'):
    '## Timeseries data',unidf
    #plot the Invoice
    if st.checkbox('Plot of Invoice over time'):
        st.line_chart(unidf)
        

#check where data is stationary
if st.sidebar.checkbox('Dicky fuller Test'):
    '## Dicky Fuller'
    X = unidf.Invoice
    result = adfuller(X)
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write('\t%s: %.3f' % (key, value))
    
#Check ACF and PACF
if st.sidebar.checkbox('ACF and PACF estimation'):
    acf_plot = plot_acf(unidf.Invoice,lags=20)
    pacf_plot = plot_pacf(unidf.Invoice,lags=20)        
    '## ACF and PACF estimation',acf_plot,pacf_plot

p= st.sidebar.slider('p',0,30,0)
d= st.sidebar.slider('d',0,3,1)
q= st.sidebar.slider('q',0,30,1)          
#ARIMA model
arima = ARIMA(unidf, order = (p,d,q))
arima_model = arima.fit()
if st.sidebar.checkbox('ARIMA summary'):
    '## ARIMA Model'
    st.text(arima_model.summary2())
    if st.checkbox('Forecast'):
        'Forecast of upcoming 3 months',arima_model.plot_predict(1, 160)
        if st.checkbox('Predicted values'):
            st.text(arima_model.predict(88,160))







