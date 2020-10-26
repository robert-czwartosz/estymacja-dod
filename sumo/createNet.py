import xml.etree.ElementTree as ET


nodes = {}
with open('map_node.txt') as file:
    lines = file.readlines()

for line in lines:
    line = line.split('\t')
    if len(line) > 2:
        try:
            nodes['n'+line[0]] = (str(int(line[1])/60), str(int(line[2])/60))
        except ValueError:
            print(nodes)
print(nodes)

rootNodes = ET.Element("nodes")
for nodID in nodes.keys():
    x, y = nodes[nodID]
    nodeElem = ET.SubElement(rootNodes, "node", 
            id=nodID, x=x, y=y, 
            type="traffic_light")





edges = []
with open('map_net.txt') as file:
    lines = file.readlines()

for line in lines:
    line = line.split('\t')
    if len(line) > 2 and line[0] != '~' and (line[2],line[1]) not in edges:
        try:
            _ = int(line[1])
            edges.append((line[1], line[2]))
        except ValueError:
            print(edges)
        
tazDict = {}
edgeList = []
rootEdges = ET.Element("edges")
rootTazs = ET.Element("additional")
rootTazs.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
rootTazs.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/additional_file.xsd')
for edge in edges:
    n1, n2 = edge
    
    n1_2 = 'n'+n1+'_'+n2
    x1_2 = float(nodes['n'+n1][0])*2/3 + float(nodes['n'+n2][0])*1/3
    y1_2 = float(nodes['n'+n1][1])*2/3 + float(nodes['n'+n2][1])*1/3
    nodeElem = ET.SubElement(rootNodes, "node", 
            id=n1_2, x=str(x1_2), y=str(y1_2), 
            type="priority")
    
    n2_1 = 'n'+n2+'_'+n1
    x2_1 = float(nodes['n'+n1][0])*1/3 + float(nodes['n'+n2][0])*2/3
    y2_1 = float(nodes['n'+n1][1])*1/3 + float(nodes['n'+n2][1])*2/3
    nodeElem = ET.SubElement(rootNodes, "node", 
            id=n2_1, x=str(x2_1), y=str(y2_1), 
            type="priority")
    
    # n1 <=======> n1_2
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to=n1_2, id=n1+'to'+n1_2[1:], 
            type="2L15")
    edgeElem.set('from', 'n'+n1)
    
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to='n'+n1, id=n1_2[1:]+'to'+n1, 
            type="2L15")
    edgeElem.set('from', n1_2)
    
    #edgeList.append(('T', n1_2, 'n'+n1, n1_2[1:]+'to'+n1))
    #edgeList.append(('T', 'n'+n1, n1_2, n1+'to'+n1_2[1:]))
    
    # n1_2 <========> n2_1
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to=n2_1, id=n1_2[1:]+'to'+n2_1[1:], 
            type="2L15")
    edgeElem.set('from', n1_2)
    
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to=n1_2, id=n2_1[1:]+'to'+n1_2[1:], 
            type="2L15")
    edgeElem.set('from', n2_1)
    
    edgeList.append(('T', n1_2, n2_1, n1_2[1:]+'to'+n2_1[1:]))
    edgeList.append(('T', n2_1, n1_2, n2_1[1:]+'to'+n1_2[1:]))
    
    # n2_1 <========> n2
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to=n2_1, id=n2+'to'+n2_1[1:], 
            type="2L15")
    edgeElem.set('from', 'n'+n2)
    
    edgeElem = ET.SubElement(rootEdges, "edge", 
             to='n'+n2, id=n2_1[1:]+'to'+n2, 
            type="2L15")
    edgeElem.set('from', n2_1)
    
    #edgeList.append(('T', n2_1, 'n'+n2, n2_1[1:]+'to'+n2))
    #edgeList.append(('T', 'n'+n2, n2_1, n2+'to'+n2_1[1:]))
    
    if n1 not in tazDict.keys():
        tazElem1 = ET.SubElement(rootTazs, "taz", id='TAZ_'+n1)
        tazDict[n1] = tazElem1
    else:
        tazElem1 = tazDict[n1]
    ET.SubElement(tazElem1, "tazSource", id=n1+'to'+n1_2[1:], weight="1.0")
    ET.SubElement(tazElem1, "tazSink", id=n1_2[1:]+'to'+n1, weight="1.0")
    if n2 not in tazDict.keys():
        tazElem2 = ET.SubElement(rootTazs, "taz", id='TAZ_'+n2)
        tazDict[n2] = tazElem2
    else:
        tazElem2 = tazDict[n2]
    ET.SubElement(tazElem2, "tazSource", id=n2+'to'+n2_1[1:], weight="1.0")
    ET.SubElement(tazElem2, "tazSink", id=n2_1[1:]+'to'+n2, weight="1.0")

treeTazs = ET.ElementTree(rootTazs)
treeTazs.write('map.taz.xml')

treeNodes = ET.ElementTree(rootNodes)
treeNodes.write('map.nod.xml')

treeEdges = ET.ElementTree(rootEdges)
treeEdges.write('map.edg.xml')

fileEdges = open('detectors\\edges.txt','w')
for e in edgeList:
    fileEdges.write(e[0]+' '+e[1]+' '+e[2]+' '+e[3]+' '+'\n')
fileEdges.close()