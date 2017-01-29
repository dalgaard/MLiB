import numpy as np
from kmeans import *


names = []
Xread = []
with open('project0-data.tsv','r') as f:
    for line in f:
        names.append(line.split()[0])
        Xread.append([float(a) for a in line.split()[1:]])
X = np.array(Xread).transpose()
#print(X)
X = normalizeColumns(X)

K=3

findClusters(X,K,plot='f')



