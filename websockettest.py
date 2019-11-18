import asyncio
import random
import websockets
import datetime
import pandas as pd  
import time 
import csv 

# import BankNiftyExecution
# from BankNiftyExecution import on_ticks

async def time(websocket, path):
    i=1
    while True:
        # dd = datetime.datetime.utcnow().isoformat() + "Z"
 
        # now=['1','2','3']
        # print (on_ticks(),"on tick")
        # now=BankNiftyExecution.on_ticks()
        # dd=[]
        # with open("finaqql.csv")as f:
        #     data = csv.reader(f)
        #     dd=data.iloc[:,-1]
        dd=[]
      
        cc=pd.read_csv("master.csv").iloc[:,i]
        i=i+1
        print(cc)
        dd=cc.to_string(index=False)
        # dd=cc.tolist()
        
        # print('ccc',end="\n")
        # print(dd)
        # for i in cc:
        #     dd.append(i)
        print("here",type(dd),dd)
        # time.sleep(.300)
        # for i in dd:
        await websocket.send(str(dd))
        await asyncio.sleep(random.random() * 3)

start_server = websockets.serve(time, "127.0.0.1", 5675)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()