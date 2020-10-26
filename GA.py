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

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from config import config
cfg = config()

from saaf import DenseSAAF


fileIn = open(cfg.InputDirHist+"\\In.pkl",'rb')
In = pickle.load(fileIn)
fileIn.close()

fileOut = open(cfg.OutputDirHist+"\\Out.pkl",'rb')
Out = pickle.load(fileOut)
fileOut.close()
#print(In[0])
LinksCounts = np.average(Out, axis=0)
InShape = np.shape(In[0])
OutShape = np.shape(Out)
delta = cfg.delta
T = cfg.N #InShape[2]
n = cfg.n
m = cfg.m
N = cfg.Noffline # rozmiar populacji
#print(LinksCounts)


fileOd = open(cfg.ODgenPatternPath,'rb')
Od = pickle.load(fileOd)
fileOd.close()
ODfrac, ODfloor = np.modf(Od)
MIN = cfg.MINprior * ODfloor - 0.001
MAX = cfg.MAXprior * ODfloor + 1
print(MIN)


def computeFitness(population):
    fitness = np.zeros(np.shape(population)[0])
    InputCNN = np.zeros((N*(T-delta),n,n,delta+1))
    linkCountsE = np.zeros((N*(T-delta),m))
    for k in range(np.shape(population)[0]):
        for i in range(delta,T):
            InputCNN[i-delta+k*(T-delta)] = population[k,:,:,i-delta:i+1]
    linkCountsE = model.predict(InputCNN)
    linkCountsE = np.array(np.split(linkCountsE,N))
    sqdiff = np.square(linkCountsE - LinksCounts[delta:])
    suma = np.abs(linkCountsE + LinksCounts[delta:])
    fitness = m/np.sum(np.sqrt(2*sqdiff/suma),axis=(1,2))
    return fitness

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
            cxpoint = random.randint(0, n*n*T)
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
    los = np.random.rand(N,n,n,T)
    population[los<mutprob] = np.random.randint(np.ones((N,n,n,T))*MAX)[los<mutprob]
    for i in range(n):
        population[:,i,i,:] = 0


if __name__ == "__main__":  # confirms that the code is under main function

    population = np.random.randint(MIN.reshape(1,n,n,T),MAX.reshape(1,n,n,T),size=(N,n,n,T))
    for i in range(n):
        population[:,i,i,:] = 0
    fitnessMaxList = []
    fitnessAvgList = []
    
    from tensorflow.keras.utils import CustomObjectScope
    with CustomObjectScope({'DenseSAAF': DenseSAAF}):
        model = load_model('Assigner.h5', compile=False)
    import time
    fitness = computeFitness(population)
    best = (population[np.argmax(fitness)], np.max(fitness))
    gc.collect()
    NGEN=cfg.NGENoffline
    crossprob = cfg.crossprob_offline
    mutprob = cfg.mutprob_offline
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
    patternOD = population[np.argmax(fitness)]
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
    print(patternOD)
    patternFile = open("patternOD.pkl",'wb')
    pickle.dump(patternOD,patternFile)
    patternFile.close()
