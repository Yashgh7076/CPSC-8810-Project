import json
import numpy as np

with open ('DL_labels.json') as json_file:
    data = json.load(json_file)
x1 = [] 
bully = []
victim = []

j = 0
#num = len(data)

for i in range(10):
    if 'Bully' in data[i]['Label']:
        list_c = data[i]['Label']['Bully']
        j = len(list_c) # No. of bullies
        for k in range(j):
            x1 = list_c[k]['geometry'][0]['x']
            y1 = list_c[k]['geometry'][0]['y']
            x2 = list_c[k]['geometry'][1]['x']
            y2 = list_c[k]['geometry'][1]['y']
            x3 = list_c[k]['geometry'][2]['x']
            y3 = list_c[k]['geometry'][2]['y']
            x4 = list_c[k]['geometry'][3]['x']
            y4 = list_c[k]['geometry'][3]['y']
            bully.append(x1)

    if 'Victim' in data[i]['Label']:
        list_c = data[i]['Label']['Victim']
        j = len(list_c) # No. of victims
        for k in range(j):            
            x5 = list_c[k]['geometry'][0]['x']
            y5 = list_c[k]['geometry'][0]['y']
            x6 = list_c[k]['geometry'][1]['x']
            y6 = list_c[k]['geometry'][1]['y']
            x7 = list_c[k]['geometry'][2]['x']
            y7 = list_c[k]['geometry'][2]['y']
            x8 = list_c[k]['geometry'][3]['x']
            y8 = list_c[k]['geometry'][3]['y']
            victim.append(x5)
        print('Bully',bully)
        print('Victim',victim)
        
   