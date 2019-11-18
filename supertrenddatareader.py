import pandas as pd 

#minute level
soopertrendata=pd.read_csv("new_supertrend2.csv")
data=pd.read_csv("sbi.csv")
opnP=data['open']
closP=data['close']
highP=data['high']
lowP=data['low']
dateT=data['datetime']
supertrenddata=soopertrendata['SuperTrend']

for  x in range(5,len(supertrenddata),15): 
    if supertrenddata[x]> closP[x+14] and supertrenddata[x-1]> closP[x-1]:
        print ( dateT[x], "green")
