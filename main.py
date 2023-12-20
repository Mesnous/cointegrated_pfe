import statsmodels.tsa.vector_ar.vecm
import statsmodels.tsa.vector_ar.var_model
import statsmodels.tsa.ar_model
import statsmodels.tsa.stattools
import pandas
import yfinance as yf
import pathlib
import datetime
import numpy as np
import arch
import matplotlib.pyplot as plt
import mgarch

def DataMAJ (Ticker) :
    #problem with time
    try :
        Data = pandas.read_csv("./T_Data/" + str(Ticker).upper() + ".csv")
    except :
        return pandas.DataFrame()
    Last_Index = Data["Date"].iloc[-1]
    if (str(Last_Index) > str(datetime.date.today() - datetime.timedelta(days=1))) & (datetime.date.today().weekday() <= 5) :
        Temp = yf.download(str(Ticker).upper(),start=Last_Index,end=str(datetime.date.today() - datetime.timedelta(days=1)))
        Data = pandas.concat([Data,Temp])
    return Data

def DataPull(Ticker_Array,DataDepth) :
    today = datetime.datetime.now() 
    delta = datetime.timedelta(days=DataDepth)
    for Ticker in Ticker_Array :
        #Data = DataMAJ(Ticker)
        if 1 :
            globals()[str(Ticker).upper()] = yf.download(str(Ticker).upper(),today-delta)
            globals()[str(Ticker).upper()].to_csv(pathlib.Path("./T_Data/"+str(Ticker).upper()+".csv"))
            """
        else :
            Data.set_index("Date",inplace=True)
            globals()[str(Ticker).upper()] = Data
            globals()[str(Ticker).upper()].to_csv(pathlib.Path("./T_Data/"+str(Ticker).upper()+".csv"),index = "Date")
            """
            
def DataProcessing(Ticker_Array) :
    if len(Ticker_Array) >=2 :
        PreProcessData = pandas.DataFrame()
        for Ticker in Ticker_Array :
            locals()[str(Ticker).upper()+"_Close"] = globals()[str(Ticker).upper()]["Adj Close"]
            locals()[str(Ticker).upper()+"_Close"] = pandas.DataFrame(locals()[str(Ticker).upper()+"_Close"].values.reshape(len(locals()[str(Ticker).upper()+"_Close"]),1))
            locals()[str(Ticker).upper()+"_Close"] = locals()[str(Ticker).upper()+"_Close"].set_index(globals()[str(Ticker).upper()].index)
            locals()[str(Ticker).upper()+"_Close"] = locals()[str(Ticker).upper()+"_Close"].rename(columns={0:str(Ticker).upper()})
        for Ticker_index in range(len(Ticker_Array)) :
            if Ticker_index == 0 :
                PreProcessData = locals()[str(Ticker_Array[Ticker_index]).upper()+"_Close"]
            else :
                PreProcessData = PreProcessData.join(locals()[str(Ticker_Array[Ticker_index]).upper()+"_Close"])
        PreProcessData = PreProcessData.dropna()
        return PreProcessData
    else :
        SectorTicker = yf.Ticker(Ticker_Array[0].upper()).info["sector"]
        print(SectorTicker)
        
def DataProcessingReturn (Data) :
    Return = pandas.DataFrame(range(len(Data[Data.columns[0]])))
    Return.index = Data.index
    for i in range(len(Data.columns)):
        Return[str(Data.columns[i]) + "_Return"] = 0
        for j in range(1,len(Data[Data.columns[i]])) :
            Return[str(Data.columns[i]) + "_Return"][j] = (Data.iloc[j,i]/Data.iloc[j-1,i])-1
    Return.drop(columns=Return.columns[0], axis=1,  inplace=True)
    Return.dropna(inplace=True)
    return Return.iloc[1:]

def DataProcessingVol (Data,Vol_Window) :
    Vol = pandas.DataFrame(range(len(Data[Data.columns[0]])))
    Vol.index = Data.index
    for i in range(len(Data.columns)):
        Vol[str(Data.columns[i]) + "_Return"] = 0
        for j in range(1,len(Data[Data.columns[i]])) :
            Vol[str(Data.columns[i]) + "_Return"][j] = (Data.iloc[j,i]/Data.iloc[j-1,i])-1
        Vol[str(Data.columns[i]) + "_Vol"] = Vol[str(Data.columns[i]) + "_Return"].rolling(window=Vol_Window).std() * np.sqrt(252/Vol_Window)
    Vol.drop(columns=Vol.columns[0], axis=1,  inplace=True)
    for i in Vol.columns :
        if "Return" in i :
            Vol.drop(columns=i, axis=1,  inplace=True)
    Vol.dropna(inplace=True)
    return Vol

############################################################################### model AR,VAR,VECM and test

def AR(Data,horizon) : 
    lag_order = statsmodels.tsa.ar_model.ar_select_order(Data, maxlag=25).ar_lags
    model = statsmodels.tsa.ar_model.AutoReg(Data, lags = lag_order)
    fited_model = model.fit()
    Pred = fited_model.predict(len(Data)-horizon)
    return Pred

def VAR(Data,horizon) :
    Train_Data = Data
    model = statsmodels.tsa.vector_ar.var_model.VAR
    fited_model = model(endog=Train_Data).fit(50)
    return fited_model.forecast(Data[-fited_model.k_ar:].values, horizon)

def VECM(Data,horizon) :
    lag = statsmodels.tsa.vector_ar.vecm.select_order(Data, maxlags=20, deterministic="n").selected_orders["aic"]
    coint_rank = statsmodels.tsa.vector_ar.vecm.select_coint_rank(Data, 0, lag)
    model = statsmodels.tsa.vector_ar.vecm.VECM(Data, deterministic="n", k_ar_diff=lag, coint_rank=coint_rank.rank)
    fited_model = model.fit()
    return fited_model.predict(horizon)

def Pred_Ping(Algo,Data,minimum_data_len,horizon,upgrade,downgrade): 
    up = []
    down = []
    for i in range (len(Data)-(minimum_data_len+horizon)) :
        if Algo == "VAR" :
            Pred = VAR(Data.iloc[:i+minimum_data_len],horizon)
        elif Algo == "VECM" :
            Pred = VECM(Data.iloc[:i+minimum_data_len],horizon)
        for j in range(len(Data.iloc[0])):
            Ratio = Pred[-1][j]/(Data.iloc[i+minimum_data_len,j])
            if Ratio > upgrade+1 : 
                up.append([i,j,Ratio])
            elif Ratio < 1-downgrade :
                down.append([i,j,Ratio])
    return Pred_Ping_Verif(Algo,up,down,horizon)

def Pred_Ping_Verif(Algo,up,down,horizon) :
    acc = []
    val = []
    if up != [] :
        for i in up :
            if ProcessedData.iloc[i[0]+horizon,i[1]] / ProcessedData.iloc[i[0],i[1]] > 1:
                acc.append(1)
            else:
                acc.append(0)
            val.append(ProcessedData.iloc[i[0]+horizon,i[1]]-ProcessedData.iloc[i[0],i[1]])
    if down != [] :
        for i in down :
            if ProcessedData.iloc[i[0]+horizon,i[1]] / ProcessedData.iloc[i[0],i[1]] < 1:
                acc.append(1)
            else:
                acc.append(0)
            val.append(-(ProcessedData.iloc[i[0]+horizon,i[1]]-ProcessedData.iloc[i[0],i[1]]))
    if down != [] and up != [] :
        acc.append(.5)
        val.append(0)
    mean_acc = np.mean(acc)
    mean_val = np.mean(val)
    print ("La precision du model " + str(Algo) + " à " + str(horizon) + " jours est de " + str(round(mean_acc*100,2)) + "% avec une pluvalue par operation de " + str(round(mean_val,2)) + "$ soit " + str(round(np.sum(val),2)) + "$")
    return mean_acc,mean_val,len(val)

def Acc_testing(Data,MinData,nDay,mini,maxi,step) :
    Var_acc = []
    Var_price = []
    Vecm_acc = []
    Vecm_price = []
    for i in range(1,nDay+1):
        accuracy = []
        price = []
        for j in np.arange(mini,maxi,step).tolist() :
            Var_Calc = Pred_Ping("VAR",Data,MinData,i,j,j)
            accuracy.append(Var_Calc[0])
            price.append(Var_Calc[1]*Var_Calc[2])
        Var_acc.append(np.mean(accuracy))
        Var_price.append(np.mean(price))
        accuracy = []
        price = []
        for j in np.arange(mini,maxi,step).tolist() :
            Vecm_Calc = Pred_Ping("VECM",Data,MinData,i,j,j)
            accuracy.append(Vecm_Calc[0])
            price.append(Vecm_Calc[1]*Vecm_Calc[2])
        Vecm_acc.append(np.mean(accuracy))
        Vecm_price.append(np.mean(price))
    return Var_acc,Var_price,Vecm_acc,Vecm_price

def RMSE (Data,minimum_data_len,horizon):
    Pred_H_VAR = []
    Pred_H_VECM = []
    Real_H = []
    for i in range (len(Data)-(minimum_data_len+horizon)) :
        Pred_VAR = VAR(Data.iloc[:i+minimum_data_len],horizon)
        Pred_VECM = VECM(Data.iloc[:i+minimum_data_len],horizon)
        Pred_H_VAR.append(Pred_VAR[-1])
        Pred_H_VECM.append(Pred_VECM[-1])
        Real_H.append(Data.iloc[i+minimum_data_len+horizon])
    RMSE_VAR = np.sqrt((np.square(np.asarray(Real_H) - np.asarray(Pred_H_VAR))).mean(axis=0))
    RMSE_VECM = np.sqrt((np.square(np.asarray(Real_H) - np.asarray(Pred_H_VECM))).mean(axis=0))
    Results = pandas.DataFrame(np.transpose([RMSE_VAR,RMSE_VECM]),columns=["RMSE_VAR","RMSE_VECM"])
    Results.index = Data.columns
    return Results
    

############################################################################### model ARCH,GARCH and test

def ARCH (ProcessedData,horizon) :
    DataReturn = DataProcessingReturn(ProcessedData) * 100
    Pred = []
    for i in DataReturn.columns :
        model = arch.arch_model(DataReturn[i],mean="Zero",vol="Arch",p=20)
        fited_model = model.fit()
        Pred.append(np.transpose(fited_model.forecast(horizon=horizon).variance[-1:]))
    Pred[0].columns = [DataReturn.columns[0] + str("_Vol")]
    for i in range(1,len(Pred)) :
        Pred[i].columns = [DataReturn.columns[i] + str("_Vol")]
        Pred[0] = Pred[0].join(Pred[i])
    return Pred[0]

def GARCH (ProcessedData,horizon) :
    DataReturn = DataProcessingReturn(ProcessedData) * 100
    Pred = []
    for i in DataReturn.columns :
        model = arch.arch_model(DataReturn[i],mean="Zero",vol="Garch",p=20,q=20)
        fited_model = model.fit()
        Pred.append(np.transpose(fited_model.forecast(horizon=horizon).variance[-1:]))
    Pred[0].columns = [DataReturn.columns[0] + str("_Vol")]
    for i in range(1,len(Pred)) :
        Pred[i].columns = [DataReturn.columns[i] + str("_Vol")]
        Pred[0] = Pred[0].join(Pred[i])
    return Pred[0]

def MGARCH (ProcessedData,horizon) :
    Pred = pandas.DataFrame()
    DataReturn = DataProcessingReturn(ProcessedData) * 100
    model = mgarch.mgarch("t")
    model.fit(DataReturn)
    for i in range(1,horizon+1) :
        Pred["Vol at " + str(i) + " Days"] = np.sqrt(np.diag(model.predict(i)['cov']))
    Pred.index = ProcessedData.columns
    return Pred.T

def Granger (Models,ProcessedData,MaxLag) :
    lags = range(1, MaxLag + 1)
    DataReturn = DataProcessingReturn(ProcessedData) * 100
    for k in Models :
        for i in range(len(DataReturn.columns)) :
            for j in DataReturn.columns :
                model = arch.arch_model(DataReturn[j],mean="Zero",vol=str(k).lower().capitalize(),p=1)
                locals()[str(j) + "_Model"] = model.fit()
                locals()[str(j) + "_resid"]  = locals()[str(j) + "_Model"].resid
            if i >= 1 :
                Granger = statsmodels.tsa.stattools.grangercausalitytests(np.column_stack(((locals()[str(DataReturn.columns[0]) + "_resid"]),locals()[str(DataReturn.columns[i]) + "_resid"])) , MaxLag, verbose=0)
                P_Val = [Granger[lag][0]['ssr_ftest'][1] for lag in lags]
                
                plt.figure(figsize=(10, 6))
                plt.plot(lags, P_Val, marker='o', linestyle='-', color='b')
                plt.xlabel('Lag')
                plt.ylabel('p-value')
                plt.title('Granger Causality Test Results between ' + str(ProcessedData.columns[0]) + " and " + str(ProcessedData.columns[i]) + " with Model " + str(k))
                plt.axhline(0.05, color='r', linestyle='-', label='Significance Level (0.05)')
                plt.legend()
                plt.show()
    return

#Faire le test de qualité des pred de vol

############################################################################### main

Ticker_Array = ["GOOG","AAPL","META","SPY"]
Model_Vol_Array = ["ARCH","GARCH"]

DataPull(Ticker_Array, 1500)
DataProcessing(Ticker_Array)

ProcessedData = DataProcessing(Ticker_Array)

ProcessedData.index = pandas.DatetimeIndex(ProcessedData.index).to_period('D')

#Train model ( not mendatory )

#for i in range(len(Ticker_Array)) :
#    globals()["AR_Trained_" + Ticker_Array[i]] = AR(ProcessedData[Ticker_Array[i]],10)

#VAR_Pred = VAR(ProcessedData,10)
#VECM_Pred = VECM(ProcessedData,10)

#ARCH_Pred = ARCH(ProcessedData,10)
#GARCH_Pred = GARCH(ProcessedData,10)
#MGARCH_Pred = MGARCH(ProcessedData,10)

#RMSE_VAR_VECM = RMSE(ProcessedData,len(ProcessedData)-100,10)
############################################################################### tests

Test_Model = Acc_testing(ProcessedData,len(ProcessedData)-100,10,.01,.05,.005)

#Granger (Model_Vol_Array,ProcessedData,10)