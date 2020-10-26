import xml.etree.ElementTree as ET
class config():
    def __init__(self):
        self.debug = True
        # Data generation
        self.processes = 6
        self.continuePrevious = False
        self.NsamplesPrior = 10
        self.NsamplesHist = 14
        self.NsamplesReal = 1
        self.dT = 15 # przedział czasu w minutach
        self.N = 24 # czas trwania symulacji jako wielokrotność dT
        self.delta = 2

        self.tazPath = 'sumo\\map.taz.xml'
        self.edgesPath = 'sumo\\detectors\\edges.txt'

        #===========================================
        tazs = ET.parse(self.tazPath).getroot()
        tazslist = tazs.findall('taz')
        self.n = len(tazslist)
        
        edges = []
        with open(self.edgesPath) as file:
            lines = file.readlines()
        for line in lines:
            line = line.split(' ')
            if len(line) > 4:
                edges.append(line[3])
        self.m = len(edges)
        #=====================================================
        self.ODgenPatternPath = 'ODgenerator_pattern.pkl'
        self.MINprior = 0.75
        self.MAXprior = 1.25
        self.MINreal = 0.9
        self.MAXreal = 1.1

        self.rootDataDir = 'Data\\'
        self.InputDirPrior = self.rootDataDir + 'InputPrior'
        self.SumoDirPrior = self.rootDataDir + 'SumoPrior'
        self.OutputDirPrior = self.rootDataDir + 'OutputPrior'

        self.InputDirHist = self.rootDataDir + 'InputHist'
        self.SumoDirHist = self.rootDataDir + 'SumoHist'
        self.OutputDirHist = self.rootDataDir + 'OutputHist'

        self.InputDirRealTime = self.rootDataDir + 'InputRealTime'
        self.SumoDirRealTime = self.rootDataDir + 'SumoRealTime'
        self.OutputDirRealTime = self.rootDataDir + 'OutputRealTime'

        self.sumoDir = 'sumo'
        self.netFile = 'map.net.xml'

        self.od2tripsOptions = ['--no-step-log', 'true','-v','false']
        self.durouterOptions = ['-X','never', '--no-step-log', 'true']
        self.sumoOptions = ['--time-to-teleport', '300', '--no-internal-links', 'true', '--collision.mingap-factor', '0.0', '--collision.action', 'remove','--collision.check-junctions', 'true', '--collision.stoptime','0', '--no-step-log','true','-W','true']
        
        # Traing CNN
        self.scaling = False
        self.epochs = 3000
        self.early_stopping = 50
        self.batch_size = 3000
        self.numCNNFilters = [2,2,2]#[256, 1024, 4096]
        # parametry warstwy perceptronowej
        self.ksi = 25
        self.eta = 2

        # Genetic Algorithm Offline
        self.Noffline = 800 # population
        self.NGENoffline = 500 # generations
        self.crossprob_offline = 0.9
        self.mutprob_offline = 0.00001
        
        # Genetic Algorithm Online
        self.Nonline = 800 # population
        self.NGENonline = 500 # generations
        self.crossprob_online = 0.9
        self.mutprob_online = 0.0001
        self.t = self.delta+1 # estymowany przedział
