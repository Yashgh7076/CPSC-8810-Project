# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:05:44 2019

@author: Yadnyesh
"""

import json

with open ('export-2019-04-16T03_44_40.557Z.json') as json_file:
    data = json.load(json_file)
    
# Bounding box co-ordinates for bully
x1 = data[0]['Label']['Bully'][0]['geometry'][0]['x']
y1 = data[0]['Label']['Bully'][0]['geometry'][0]['y']

x2 = data[0]['Label']['Bully'][0]['geometry'][1]['x']
y2 = data[0]['Label']['Bully'][0]['geometry'][1]['y']

x3 = data[0]['Label']['Bully'][0]['geometry'][2]['x']
y3 = data[0]['Label']['Bully'][0]['geometry'][2]['y']

x4 = data[0]['Label']['Bully'][0]['geometry'][3]['x']
y4 = data[0]['Label']['Bully'][0]['geometry'][3]['y']


# Bounding box co-ordinates for victim
x5 = data[0]['Label']['Victim'][0]['geometry'][0]['x']
y5 = data[0]['Label']['Victim'][0]['geometry'][0]['y']

x6 = data[0]['Label']['Victim'][0]['geometry'][1]['x']
y6 = data[0]['Label']['Victim'][0]['geometry'][1]['y']

x7 = data[0]['Label']['Victim'][0]['geometry'][2]['x']
y7 = data[0]['Label']['Victim'][0]['geometry'][2]['y']

x8 = data[0]['Label']['Victim'][0]['geometry'][3]['x']
y8 = data[0]['Label']['Victim'][0]['geometry'][3]['y']
    

    
