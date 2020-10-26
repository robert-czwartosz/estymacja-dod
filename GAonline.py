#import multiprocessing
#from multiprocessing import Lock, Process, Queue, current_process
import random, pickle
import numpy as np
import gc
import tensorflow
import math
import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

import matplotlib.pyplot as plt
import statsmodels.api as sm

from tensorflow.keras.models import load_model

from saaf import DenseSAAF

from config import config
cfg = config()

#===================================================
# Wczytanie danych
#================================================
patternFile = open("patternOD.pkl",'rb')
patternOD = pickle.load(patternFile)
patternFile.close()

fileIn = open(cfg.InputDirRealTime+"\\In0\\In0.pkl",'rb')
In = pickle.load(fileIn)
fileIn.close()

fileOut = open(cfg.OutputDirRealTime+"\\Out0.pkl",'rb')
Out = pickle.load(fileOut)
fileOut.close()
print(patternOD[:,:,0])
print(patternOD[:,:,1])
print(patternOD[:,:,2])
print(patternOD[:,:,3])
#exit()
InShape = np.shape(patternOD)
OutShape = np.shape(Out)
delta = cfg.delta
T = cfg.N #InShape[2]
n = cfg.n #InShape[0]
m = cfg.m #OutShape[1]

dT = str(cfg.dT)
N = cfg.Nonline # rozmiar populacji
t = delta
LinksCounts = Out[t]
RealOD = In[:,:,t]

MIN = patternOD[:,:,t-delta:t+1] * 0.9 - 0.001
MAX = patternOD[:,:,t-delta:t+1] * 1.1 + 1
#=======================================================
# Funkcje dla algorytmu genetycznego
#====================================================

def computeFitness(population):
    fitness = np.zeros(N)
    InputCNN = np.zeros((N,n,n,delta+1))
    linkCountsE = np.zeros((N,m))
    for k in range(N):
        InputCNN[k] = population[k,:,:,t-delta:t+1]
    linkCountsCNN = model.predict(InputCNN)
    sqdiff = np.square(linkCountsCNN - LinksCounts)
    suma = np.abs(linkCountsCNN + LinksCounts)
    sqdiff[:,LinksCounts < 2] = 0
    fitness = m/np.sum(np.sqrt(2*sqdiff/suma),axis=1)
    #print(fitness)
    return fitness

def MAEgt5(OD):
    return np.average(np.abs(RealOD - OD)[RealOD > 5])
def MAE(OD):
    return np.average(np.abs(RealOD - OD))

def MAPEgt5(OD):
    return 100*np.average((np.abs(RealOD - OD)/np.maximum(RealOD,1))[RealOD > 5])
def MAPE(OD):
    return 100*np.average(np.abs(RealOD - OD)/np.maximum(RealOD,1))

def R2(Y_real, Y_pred):
    return 1 - np.sum((Y_pred - Y_real)**2)/np.sum((Y_real - np.average(Y_real))**2)

def select(fitness):
    probs = (fitness - np.min(fitness)) / np.sum(fitness - np.min(fitness))
    #probs = fitness / np.sum(fitness)
    elements = list(range(N))
    probabilities = list(probs)
    return np.random.choice(elements, int(N/2)*2, p=probabilities).reshape(int(N/2),2)

def crossover1P(choices, crossprob):
    global population
    result = np.array(population, copy=True)
    for i in range(np.shape(choices)[0]):
        c1, c2 = choices[i,0], choices[i,1]
        result[2*i] = population[c2]
        result[2*i+1] = population[c1]
        if random.random() < crossprob:
            cxpoint = random.randint(0, n*n*(delta+1))
            result[2*i].reshape(-1)[cxpoint:] = population[c1].reshape(-1)[cxpoint:]
            
            result[2*i+1].reshape(-1)[cxpoint:] = population[c2].reshape(-1)[cxpoint:]
    population = result
    
def crossover2P(choices, crossprob):
    global population
    result = np.array(population, copy=True)
    for i in range(np.shape(choices)[0]):
        c1, c2 = choices[i,0], choices[i,1]
        result[2*i] = population[c2]
        result[2*i+1] = population[c1]
        if random.random() < crossprob:
            cxpoint1 = [random.randint(0, n-1),random.randint(0, n-1),random.randint(0, T-1)]
            cxpoint2 = [random.randint(0, n),random.randint(0, n),random.randint(0, T)]
            for j in range(3):
                if cxpoint1[j] >= cxpoint2[j]:
                    tmp = cxpoint1[j]
                    cxpoint1[j] = cxpoint2[j]
                    cxpoint2[j] = tmp + 1
            cxpoint1 = tuple(cxpoint1)
            cxpoint2 = tuple(cxpoint2)
            result[2*i,cxpoint1[0]:cxpoint2[0],cxpoint1[1]:cxpoint2[1],cxpoint1[2]:cxpoint2[2]] = \
            population[c1,cxpoint1[0]:cxpoint2[0],cxpoint1[1]:cxpoint2[1],cxpoint1[2]:cxpoint2[2]]
            
            result[2*i+1,cxpoint1[0]:cxpoint2[0],cxpoint1[1]:cxpoint2[1],cxpoint1[2]:cxpoint2[2]] = \
            population[c2,cxpoint1[0]:cxpoint2[0],cxpoint1[1]:cxpoint2[1],cxpoint1[2]:cxpoint2[2]]
    population = result

def mutate(mutprob):
    global population
    los = np.random.rand(N,n,n,delta+1)
    population[los<mutprob] = np.random.randint(MIN,MAX,size=(N,n,n,delta+1))[los<mutprob]
    for i in range(n):
        population[:,i,i,:] = 0

#==============================================
# Wyświetlanie wykresów
#===============================================

def ToStr(Th, Tmin):
    if Tmin < 10:
        return str(Th)+'.0'+str(Tmin)
    return str(Th)+':'+str(Tmin)

def showFlowPlot(i, OD):
    T0 = 7 * 60 + i*int(dT)
    T0h = int(T0/60)
    T0min = int(T0%60)

    T1 = T0 + int(dT)
    T1h = int(T1/60)
    T1min = int(T1%60)


    X = OD.flatten()
    Y = RealOD.flatten()
    results = sm.OLS(Y,sm.add_constant(X)).fit()

    print(results.summary())

    plt.scatter(X,Y)

    plt.plot(X, results.fittedvalues,color='orange')
    plt.plot(results.fittedvalues, results.fittedvalues,color='green')

    plt.title('Comparisons of the estimated and observed OD flows \n during TI'+str(i+1)+'('+ToStr(T0h,T0min)+' AM ~ '+ToStr(T1h,T1min)+' AM)')
    plt.xlabel('Estimated Flow (veh/'+dT+' min)')
    plt.ylabel('Observed Flow (veh/'+dT+' min)')

    a = '1.00'
    b = '0.00'
    s = 'asd'
    if len(results.params) == 2:
        a = str(round(results.params[1], 4))
        if results.params[0] < 0:
            b = '- '+str(round(abs(results.params[0]), 4))
        else:
            b = '+ '+str(round(abs(results.params[0]), 4))
    else:
        a = str(results.params[0])
    s = 'y = '+a+'x '+b+'\n'+\
            r'$R^2$ = '+str(round(results.rsquared, 4))
    plt.text(0.1*np.max(X), 0.9*np.max(Y), s, multialignment="center")

    plt.show()

#=======================================================
# Algorytm genetyczny
#===================================================
if __name__ == "__main__":
    population = np.random.randint(MIN,MAX,size=(N,n,n,delta+1))
    for i in range(n):
        population[:,i,i,:] = 0

    from tensorflow.keras.utils import CustomObjectScope
    with CustomObjectScope({'DenseSAAF': DenseSAAF}):
        model = load_model('Assigner.h5', compile=False)
    import time
    fitness = computeFitness(population)
    best = (population[np.argmax(fitness)], np.max(fitness))
    gc.collect()
    NGEN = cfg.NGENonline
    crossprob = cfg.crossprob_online
    mutprob = cfg.mutprob_online
    fitnessAvgList = []
    fitnessMaxList = []
    for gen in range(NGEN):
        start_time = time.time()
        if gen % 10 == 0:
            print('Iteration '+str(gen))
        fitness = computeFitness(population)
        choices = select(fitness)
        crossover1P(choices, crossprob)
        mutate(mutprob)
        if np.max(fitness) > best[1]:
            best = (population[np.argmax(fitness)], np.max(fitness))
        if gen % 10 == 0:
            print('Time: ',time.time() - start_time)
            print('Memory usage[MB]: ',round(process.memory_info().rss/10e6))
            print('Max. fitness: ',np.max(fitness))
            print('Avg. fitness: ',np.average(fitness))
            print('Best fitness: ',best[1])
        fitnessAvgList.append(np.average(fitness))
        fitnessMaxList.append(np.max(fitness))
        gc.collect()
    ODmat = population[np.argmax(fitness)]
    plt.scatter(list(range(NGEN)),fitnessAvgList)
    plt.title('Avg. fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. fitness')
    plt.show()

    plt.scatter(list(range(NGEN)),fitnessMaxList)
    plt.title('Max. fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Max. fitness')
    plt.show()
    print(ODmat)
    print('MAE: ',MAE(ODmat[:,:,-1]),'('+str(MAEgt5(ODmat[:,:,-1]))+')')
    print('MAPE: ',MAPE(ODmat[:,:,-1]),'('+str(MAPEgt5(ODmat[:,:,-1]))+')')
    print('R^2: ', R2(RealOD, ODmat[:,:,-1]))
    showFlowPlot(t,ODmat[:,:,-1])
