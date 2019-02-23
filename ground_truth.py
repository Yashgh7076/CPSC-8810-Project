# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:03:11 2019

@author: Yadnyesh
"""

import numpy as np

# Enter the total number of images in each folder
N = 142
value = 8 # Create categorical label as per list given below

# Create labels of all 1's
# Category 5 (punching) contains fewer bullying images hence np.zeros used

L1 = np.ones(shape = [N,1], dtype = np.uint8)
#L1 = np.zeros(shape = [N,1], dtype = np.uint8)

# Create locations of label mismatch
zeros = [1, 2, 3, 5, 10, 12, 14, 19, 20, 25,
         26, 31, 37, 40, 42, 44, 51, 52, 53, 
         54, 57, 58, 66, 72, 73, 74, 75, 87,
         91, 93, 95, 100, 111, 112, 118, 119,
         120, 121, 129, 130, 131, 133, 137]

#ones = [1, 11, 12, 14, 16, 17, 20, 23, 24, 29, 30, 31, 32, 33, 36,
#        37, 38, 39, 40, 45, 52, 54, 56, 57, 58, 59, 62, 66, 77, 79,
#        80, 81, 82, 84, 85, 87, 88, 91, 98, 99, 101, 102, 103, 104,
#        107, 112, 115, 116, 117, 118, 121, 122, 123, 124, 127, 130,
#        131, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144,
#        146, 147, 148, 149, 151, 154, 155, 156, 160, 163, 164, 168,
#        169, 170, 172, 175, 176, 177, 183, 185, 187, 189, 192, 193,
#        195, 198, 201, 202, 207, 209, 218, 220, 221, 224, 229, 230,
#        237, 239, 240, 242, 245, 246, 247, 250, 252, 254, 255, 256,
#        259, 260, 261, 262, 263, 267, 268, 269, 270, 272, 273, 274,
#        277, 283, 284, 289, 291, 296, 302, 305, 312, 314, 315, 317,
#        318, 319, 321, 322, 323, 324, 325, 327, 328, 329, 335, 339,
#        340, 341, 342, 343, 345, 348, 350, 351, 352, 356, 357, 359,
#        360, 361, 363]
# Set the locations to zeros

# Bullying Vs Non - Bullying Labels
# 1 => Bullying // 0 => Non - bullying
L1[zeros] = 0

# Categorical labels
# 1 -> gossipping
# 2 -> isolation
# 3 -> laughing
# 4 -> pulling hair
# 5 -> punching
# 6 -> quarrel
# 7 -> slapping
# 8 -> stabbing
# 9 -> strangle
# 10 -> Other (non - bullying)

L2 = np.full((N, 1), value, dtype = np.uint8)
L2[zeros] = 10
 
np.savetxt('F:\Clemson University\ECE 8810_Deep Learning\Project\stabbing\labelling_bullying.txt',L1, fmt = '%1.u')
np.savetxt('F:\Clemson University\ECE 8810_Deep Learning\Project\stabbing\labelling_categories.txt',L2,fmt = '%1.u')
