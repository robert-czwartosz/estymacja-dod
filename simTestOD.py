import random
import os
import subprocess
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

from config import config
cfg = config()
DEBUG = cfg.debug

continuePrevious = cfg.continuePrevious
NsamplesPrior = cfg.NsamplesPrior
dT = cfg.dT # przedział czasu w minutach
N = cfg.N # czas trwania symulacji jako wielokrotność dT

tazPath = cfg.tazPath
os.chdir(r'.\sumo\detectors')
subprocess.call(['python', 'detector.py' ,str(dT*60)], shell=False)
os.chdir(r'..\..')

edges = []
with open(cfg.edgesPath) as file:
    lines = file.readlines()
edgesSource = []
ifedgesSource = []
edgesSink = []
ifedgesSink = []
for line in lines:
    line = line.split(' ')
    if len(line) > 4:
        edges.append(line[3])
        if line[0] == 'Source':
            edgesSource.append(line[3])
            ifedgesSource.append(True)
        else:
            ifedgesSource.append(False)
        if line[0] == 'Sink':
            edgesSink.append(line[3])
            ifedgesSink.append(True)
        else:
            ifedgesSink.append(False)
m = len(edges)
if DEBUG:
    print('edges ',edges)
    print('ifedgesSource', ifedgesSource)
    print('edgesSource', edgesSource)

def ToStr(Th, Tmin):
    if Tmin < 10:
        return str(Th)+'.0'+str(Tmin)
    return str(Th)+'.'+str(Tmin)



tazs = ET.parse(tazPath).getroot()
tazslist = tazs.findall('taz')
n = len(tazslist)

OD_legnicka_csv = np.genfromtxt('OD.txt', delimiter=',',dtype=str)
IDlist = list(OD_legnicka_csv[0])
OD_index = {}

for idx,node in enumerate(tazslist):
    nodeID = node.get('id')
    ID = nodeID.split('_')[-1]
    OD_index[idx] = IDlist.index(ID)
    OD_legnicka_csv[0][idx+1]
OD_legnicka = np.zeros((n,n))
OD_legnicka_csv = np.genfromtxt('OD.txt', delimiter=',')
for oidx,o in enumerate(tazslist):
    for didx,d in enumerate(tazslist):
        OD_legnicka[oidx][didx] = OD_legnicka_csv[OD_index[oidx]][OD_index[didx]]
OD_legnicka /= 60*6 / dT
OD_legnicka /= 5
ODfrac, ODfloor = np.modf(OD_legnicka)

odFile = open(cfg.ODgenPatternPath, 'wb')
pickle.dump(OD_legnicka, odFile)
odFile.close()

ODfrac, ODfloor = np.modf(OD_legnicka)
MIN = None
MAX = None
InputDir = None
SumoDir = None
OutputDir = None
def init(InDir, SumDir, OutDir, MINfrac, MAXfrac):
    global InputDir
    global OutputDir
    global SumoDir
    InputDir = InDir
    OutputDir = OutDir
    SumoDir = SumDir
    
    global ODfrac, ODfloor, MAX, MIN
    MIN = MINfrac * ODfloor
    MAX = MAXfrac * ODfloor + 1.0
#exit()
netFile = cfg.netFile
od2tripsOptions = cfg.od2tripsOptions
durouterOptions = cfg.durouterOptions
sumoOptions = cfg.sumoOptions
    
def plotOD(SumOD, SumDetectorSource):
    ind = np.arange(N)    # the x locations for the groups
    
    p1 = plt.bar(ind, SumOD)
    p2 = plt.bar(ind, SumDetectorSource)

    plt.ylabel('Cars')
    plt.title('OD demand stats')
    plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24'))
    #plt.yticks(np.arange(0, np.max(SumOD)+10, np.max(SumOD)/8))
    plt.legend((p1[0], p2[0]), ('OD demand', 'Departed cars'))

    plt.show()

def generateSample(sample):
    global OutputDir, InputDir, SumoDir, ODfrac, ODfloor, MIN, MAX, n, N, m
    InDir = InputDir+'\\In'+str(sample)
    SimDir = SumoDir+'\\SUMO'+str(sample)+'\\'
    os.mkdir(InDir)
    os.mkdir(SimDir)
    if DEBUG:
        print('sample', sample)
    InputSample = np.zeros((n,n,N))
    SumaOD = np.zeros(N)
    OutputSample = np.zeros((N,m))
    
    #shutil.copytree(cfg.sumoDir, SimDir)
    subprocess.call(['xcopy', cfg.sumoDir, SimDir,'/s'], shell=True)
    
    # Create OD files
    T0min = 0 # czas początkowy w minutach
    T0h = 0 # czas początkowy w godzinach
    T1min = 0 # czas końcowy w minutach
    T1h = 0 # czas końcowy w godzinach
    files = ''
    InputOD = np.zeros((n,n,N))
    for i in range(N):
        InputOD[:,:,i] = np.random.randint(MIN, MAX) * (ODfloor > 0)
    for i in range(N):
        T1min += dT
        T1h += int(T1min/60)
        T1min -= (int(T1min/60)) * 60

        files += InDir+'\\OD'+str(i)+'.od,'
        file = open(InDir+'\\OD'+str(i)+'.od', 'w')
        file.write('$O;D2\n')
        file.write('* From-Time\tTo-Time\n')
        file.write(ToStr(T0h, T0min)+' '+ToStr(T1h, T1min)+'\n')
        file.write('* Factor\n')
        file.write('1.00\n')
        file.write('*\n')
        file.write('* some\n')
        file.write('* additional\n')
        file.write('* comments\n')
        file.write('* \tFROM\tTO\tCOUNT\n')
        
        for oidx,o in enumerate(tazslist):
            for didx,d in enumerate(tazslist):
                if oidx==didx:
                    InputSample[oidx][didx][i] = 0
                    continue
                count = InputOD[oidx,didx,i] + np.random.choice(np.arange(2), p=[1-ODfrac[oidx,didx], ODfrac[oidx,didx]])
                InputSample[oidx][didx][i] = count
                file.write('\t'+o.get('id')+'\t\t\t\t'+
                d.get('id')+'\t\t\t'+str(int(count))+'\n')
        file.close()
        T0min = T1min
        T0h = T1h
    SumaOD = np.sum(InputSample, axis=(0,1))
    # Save dynamic OD to pkl file
    inFile = open(InDir+'\\In'+str(sample)+'.pkl', 'wb')
    pickle.dump(InputSample, inFile)
    inFile.close()
    # Generate output trace
    subprocess.call(['od2trips','-l', SimDir+'od2tripslog.txt', '-n', tazPath, '-d', files[:-1], '-o', SimDir+'od_file.odtrips.xml']+od2tripsOptions, shell=True)
    subprocess.call(['duarouter','-n',SimDir+netFile,'--route-files', SimDir+'od_file.odtrips.xml','-o',SimDir+'od_route_file.odtrips.rou.xml','-l', SimDir+'dualog.txt']+durouterOptions, shell=True)
    subprocess.call(['sumo-gui', '-a', SimDir+'detectors\\e1.add.xml','-e',str(N*dT*60),'-n',SimDir+netFile,'-r', SimDir+'od_route_file.odtrips.rou.xml','-l', SimDir+'sumolog.txt']+sumoOptions, shell=True)
    

    # Get link counts and turning movement flows
    linkFlow = {}
    fcd = ET.parse(SimDir+'detectors\\e1output.xml').getroot()
    for link in  fcd.findall('interval'):
        tn = int(float(link.get('begin'))/(dT*60))
        edge = link.get('id')[6:-2]
        try:
            linkFlow[(tn, edge)] += float(link.get('nVehContrib'))
        except KeyError:
            linkFlow[(tn, edge)] = float(link.get('nVehContrib'))
    
    for tn in range(N):
        for i,edge in enumerate(edges):
            OutputSample[tn][i] = linkFlow[(tn, edge)]
    if DEBUG:
        print('OutputSample*np.array(ifedgesSource)', OutputSample*np.array(ifedgesSource))
    SumaDetectorSource = np.sum(OutputSample*np.array(ifedgesSource), axis=1)
    SumaDetectorSink = np.sum(OutputSample*np.array(ifedgesSink), axis=1)
    # Save link flows to pkl file
    outFile = open(OutputDir+'\\Out'+str(sample)+'.pkl', 'wb')
    pickle.dump(OutputSample, outFile)
    outFile.close()
    #shutil.rmtree(SimDir+'cfg', ignore_errors=True)
    os.system('del '+SimDir+'*.xml')
    if DEBUG:
        print('SumaOD: ',SumaOD)
        print('SumaDetectorSource: ',SumaDetectorSource)
        print('SumaDetectorSink: ',SumaDetectorSink)
        print('SumaOD - SumaDetectorSource: ',SumaOD - SumaDetectorSource)
        print('SumaDetectorSource - SumaDetectorSink: ',SumaDetectorSource - SumaDetectorSink)
    plotOD(SumaOD, SumaDetectorSource)
    plotOD(SumaOD.cumsum(), SumaDetectorSource.cumsum())
    return InputSample, OutputSample

def generateData(InputDir, SumoDir, OutputDir, MINfrac, MAXfrac, Nsamples,cont, processes=None):
    init(InputDir, SumoDir, OutputDir, MINfrac, MAXfrac)
    begin = 0
    if cont==False:
        shutil.rmtree(InputDir, ignore_errors=True)
        shutil.rmtree(SumoDir, ignore_errors=True)
        shutil.rmtree(OutputDir, ignore_errors=True)
        os.mkdir(InputDir)
        os.mkdir(SumoDir)
        os.mkdir(OutputDir)
    else:
        InputFile0 = open(InputDir+'\\In.pkl', 'rb')
        Input0 = pickle.load(InputFile0)
        InputFile0.close()
        OutputFile0 = open(OutputDir+'\\Out.pkl', 'rb')
        Output0 = pickle.load(OutputFile0)
        OutputFile0.close()
        begin = np.shape(Input0)[0]
    with Pool(initializer = init, initargs = (InputDir, SumoDir, OutputDir, MINfrac, MAXfrac)) as pool:
        M = pool.starmap(generateSample, zip(range(begin,Nsamples+begin)))
    Input, Output = zip(*M)
    Input = np.array(Input)
    Output = np.array(Output)
    if cont:
        Input = np.concatenate((Input0,Input),axis=0)
        Output = np.concatenate((Output0,Output),axis=0)
    
    # Save all OD flows to pkl file
    inFile = open(InputDir+'\\In.pkl', 'wb')
    pickle.dump(Input, inFile)
    inFile.close()
    # Save all link counts to pkl file
    outFile = open(OutputDir+'\\Out.pkl', 'wb')
    pickle.dump(Output, outFile)
    outFile.close()

if __name__=="__main__":
    generateData(cfg.InputDirRealTime,cfg.SumoDirRealTime,cfg.OutputDirRealTime, cfg.MINreal, cfg.MAXreal, cfg.NsamplesReal, False)