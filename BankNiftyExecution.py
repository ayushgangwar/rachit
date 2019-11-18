import logging
from kiteconnect import KiteTicker
import logging
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import time
import re
import logging
from datetime import datetime
import pandas as pd
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import random
from twilio.rest import Client


instrument_tokens_to_subscribe = []
main_user = None
all_other_users = dict()
bn_30_day_spot = None
bn_today_open = None
bnf_past_day = None
kws = None
quantity = 0
web_socket = None

openP=list()
highP=list()
lowP=list()
closP=list()
volP=list()
# ltp=list()
openprice=list()
closeprice=list()
highprice=list()
lowprice=list()
volume=list()



Api_key="z1pnpfjfnmlvn0u3"
token="Wyb2JyvU4qfipedFtz9J1mf767xNivR0"
API_secret="mbyui332c95cpaun5f2mqm73c7pmvkha"
path="/home/kuants/raja/live.csv"

kite = KiteConnect(api_key="z1pnpfjfnmlvn0u3")
print(kite.login_url())
data = kite.generate_session("Pf1lHgcRjtIxlSZ8cfgs0pJHANBWkb2i", api_secret=API_secret)

kws = KiteTicker(Api_key, data['access_token'])
global data1

def historical_data():
    # b=kite.historical_data(4708097, "2019-01-01 09:15:00", "2019-01-31 15:30:00", "minute")
    print("inside historical")
    b=pd.DataFrame(kite.historical_data(12580610, "2019-09-01 09:15:00", "2019-10-30 15:30:00", "minute", continuous=1))
    b.to_csv("indusind_Sept.csv")


def place_order():
     order_id = kite.place_order(exchange=kite.EXCHANGE_NSE,
                                order_type=kite.ORDER_TYPE_LIMIT,
                                tradingsymbol='SBIN',
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=volume,
                                price=entry,
                                squareoff=target,
                                stoploss=stoploss,
                                validity=kite.VALIDITY_DAY,
                                variety=kite.VARIETY_BO,
                                product=kite.PRODUCT_MIS)

def data_input():
    historical_data()
    inputdata=pd.read_csv(path)
    current_tick= inputdata['last_price']
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    timereq="0"
    current_tick.to_csv("currenttic.csv")


def create_candle(datas,frequency):
    for x in range(0,len(datas['open'])-frequency,frequency):
        openprice.append((datas['open'][x]))
        closeprice.append((datas['close'][x+frequency]))
        highprice.append((np.max(datas['high'][x:x+frequency])))
        lowprice.append((np.min(datas['low'][x:x+frequency])))
        volume.append((np.sum(datas['volume'][x:x+frequency])))

    final_candle=pd.DataFrame({"open":openprice,"high":highprice,"low":lowprice,"close":closeprice,"volume":volume})
    return final_candle


    
def ST(df,f,n): #df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    #Calculation of ATR
    print(df)
    df['high']=df['high'].astype(float)
    df['low']=df['low'].astype(float)
    df['close']=df['close'].astype(float)
    df['volume']=df['volume'].astype(float)
    df['H-L']=abs(df['high']-df['low']).astype(float)
    df['H-PC']=abs(df['high']-df['close'].shift(1)).astype(float)
    df['L-PC']=abs(df['low']-df['close'].shift(1)).astype(float)
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1).astype(float)
    df['ATR']=np.nan
    df.ix[n-1,'ATR']=df['TR'][:n-1].mean() #.ix is deprecated from pandas verion- 0.19
    for i in range(n,len(df)):
        df['ATR'][i]=(df['ATR'][i-1]*(n-1)+ df['TR'][i])/n

    #Calculation of SuperTrend
    df['Upper Basic']=(df['high']+df['low'])/2+(f*df['ATR']).astype(float)
    df['Lower Basic']=(df['high']+df['low'])/2-(f*df['ATR']).astype(float)
    df['Upper Band']=df['Upper Basic']
    df['Lower Band']=df['Lower Basic']
    for i in (n,len(df)-1):
        if df['close'][i-1]<=df['Upper Band'][i-1]:
            if df['Upper Basic'][i]< df['Upper Band'][i-1]:
                df['Upper Band'][i]=df['Upper Basic'][i]
            else: 
                df['Upper Band'][i]=df['Upper Band'][i-1]
        else:
            print(df['Upper Basic'][i-1],"asdsadsad***********")    
            df['Upper Band'][i]=df['Upper Basic'][i]    
    for i in range(n,len(df)-1):
        if df['close'][i-1]>=df['Lower Band'][i-1]:
            df['Lower Band'][i]=max(df['Lower Basic'][i],df['Lower Band'][i-1])
        else:
            df['Lower Band'][i]=df['Lower Basic'][i]   
    df['SuperTrend']=np.nan
    for i in df['SuperTrend']:
        if df['close'][n-1]<=df['Upper Band'][n-1]:
            df['SuperTrend'][n-1]=df['Upper Band'][n-1]
        elif df['close'][n-1]>df['Upper Band'][i]:
            df['SuperTrend'][n-1]=df['Lower Band'][n-1]
    for i in range(n,len(df)-1):
        if df['SuperTrend'][i-1]==df['Upper Band'][i-1] and df['close'][i]<=df['Upper Band'][i]:
            df['SuperTrend'][i]=df['Upper Band'][i]
        elif  df['SuperTrend'][i-1]==df['Upper Band'][i-1] and df['close'][i]>=df['Upper Band'][i]:
            df['SuperTrend'][i]=df['Lower Band'][i]
        elif df['SuperTrend'][i-1]==df['Lower Band'][i-1] and df['close'][i]>=df['Lower Band'][i]:
            df['SuperTrend'][i]=df['Lower Band'][i]
        elif df['SuperTrend'][i-1]==df['Lower Band'][i-1] and df['close'][i]<=df['Lower Band'][i]:
            df['SuperTrend'][i]=df['Upper Band'][i]
    # calculate(df)
    return df

def vwap(data, period, frequency):
    datas=candledata(data, frequency)
    vwap=0
    vol_sum=0
    for i in period:
        vwap= vwap+datas['volume']*datas['close']
        vol_sum=vol_sum+ datas['volume']
        vwap=vwap/vol_sum


def sendmessage():
    account_sid = 'AC08b30813340b59d07663c3180627c272'
    auth_token = 'a5507bb530960d78e55e3f89121cf461'
    client = Client(account_sid, auth_token)

    message = client.messages \
                .create(
                     body="RBLBANK triggered"+ str(datetime.now()),
                     from_='+12018347323',
                     to='+919910795859'
                 )

    print("mesage sent")    
    

def allfunctions():
    #candle15
    datas=create_candle(data,15)
    ST=ST(datas)
    #candle10
    datas=create_candle(data,10)
    ST=ST(datas)
    #candle5
    datas=create_candle(data,5)
    ST=ST(datas)
    #vwap
    vwaps=vwap()
    #ltp


def calculate(data2):
    print("Supertrend", data2["SuperTrend"][len(data2["SuperTrend"]) -3 ] )
    print("supertrend",data2["SuperTrend"][len(data2["SuperTrend"]) -1 ])
    print("closeprice",data2["close"][len(data2["SuperTrend"]) -1 ])
    if data2["SuperTrend"][len(data2["SuperTrend"]) -3 ] > data2["close"][len(data2["SuperTrend"]) -3 ] and data2["SuperTrend"][len(data2["SuperTrend"]) -2 ] < data2["close"][len(data2["SuperTrend"]) -2 ]:
        sendmessage()
        print("message called")
    print("returned from calculate function")
ltp=[]
openP=[]
supertrend15=[]
highP=[]
lowP=[]
closP=[]
volP=[]
value_to_be_appended=0
def on_ticks(ws, ticks):
    global bnf_1_minute
    global ltp
    volume=0
    b=0
    ltp.append(float(ticks[0]['last_price']))  
    ltp2=pd.DataFrame(ltp)
    ltp2=pd.DataFrame("ltp.csv")
    if (str(datetime.now())[18]) =='0':
        openP.append(str(ltp[0]))
        highP.append(str(np.max(ltp)))
        lowP.append(str(np.min(ltp)))
        closP.append(str(ltp[len(ltp)-1]))  
        volP.append(volume)
       
        final=pd.DataFrame({"open":openP,"high":highP,"low":lowP,"close":closP,"volume":volP})
        print("minute data created",final)
        if len(openP) > 15:
            print("in if function************",len(final['open']))
            
            candle_data=create_candle(final,15)
            print("candle created and sent to st",candle_data)
            
            supertrends=ST(candle_data,3,8)
            value_to_be_appended=supertrends['SuperTrend'][len(supertrends['SuperTrend'])-2]
            supertrend15.append(value_to_be_appended)
        else:
            supertrend15.append(len(supertrend15)-1)
        final2=pd.DataFrame({"open":openP,"high":highP,"low":lowP,"close":closP,"ltp":float(ltp[len(ltp)-1]),"Supertrend15":supertrend15})
        final_csv=pd.DataFrame(final2)
        final_csv=final_csv.transpose()
        final_csv.to_csv("finaqql.csv")
        ltp=[]
        return final
    elif (len(openP)>1):      
        print("sent to csv")
        openP.append(openP[len(openP)-1])
        highP.append(highP[len(highP)-1])
        lowP.append(lowP[len(lowP)-1])
        closP.append(closP[len(closP)-1])  
        volP.append(volume)
        if len(supertrend15)==0:
            supertrend15.append(0)
        else:
            supertrend15.append(len(supertrend15)-1)
        volume=0
      
        final2=pd.DataFrame({"open":openP,"high":highP,"low":lowP,"close":closP,"ltp":float(ltp[len(ltp)-1]),"Supertrend15":supertrend15})
        final_csv=pd.DataFrame(final2)
        final_csv=final_csv.transpose()
        final_csv.to_csv("finaqql.csv")
        # ltp=[]
        # return final
    print("lop")
    return 0


def on_connect(ws, response):

    ws.subscribe([4708097])
    ws.set_mode(ws.MODE_LTP, [4708097])

def on_close(ws, code, reason):
    ws.stop()


kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close=on_close
kws.connect()

# ltpdata=pd.read_csv("ltptest.csv")
# # print(ltpdata)
# datas=ltpdata['ltp']
# for x in range(0,len(datas)):
#     time.sleep(1)
#     print(datas[x], "ticks sent")
#     on_ticks(datas[x])
#     print("bacl")