import xml.etree.ElementTree as ET

import argparse

parser = argparse.ArgumentParser(description='Create detectors.')
parser.add_argument('freq', metavar='F', type=str, 
                    help='frequency of detectors')

#freq = "900"
args = parser.parse_args()
freq = args.freq

edges = {}
with open('edges.txt') as file:
    lines = file.readlines()

for line in lines:
    line = line.split(' ')
    if len(line) > 3:
        edges[line[3]] = line[0]




root = ET.Element("additional")
root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/additional_file.xsd')
lanes = {}
net = ET.parse('..\\map.net.xml').getroot()
for edge in net.findall('edge'):
    
    if edge.get('id') in list(edges.keys()):
        #print(edge.get('id'))
        Type = edges[edge.get('id')]
        if Type == 'Source':
            pos = '-0.00001'
        elif Type == 'Sink':
            pos = '0.00001'
        elif Type == 'T':
            pos = '0.0'
        else:
            print('Error')
        for lane in edge.findall('lane'):
            laneID = lane.get('id')
            e1Detector = ET.SubElement(root, "e1Detector", 
            file="e1output.xml", freq=freq, friendlyPos="x", 
            id="e1det_"+laneID, lane=laneID, pos=pos)


tree = ET.ElementTree(root)
tree.write('e1.add.xml')