import numpy as np
from kmeans import *


names = []
Xread = []
with open('old_faithful.txt','r') as f:
    for line in f:
        names.append(line.split()[0])
        Xread.append([float(a) for a in line.split()[1:]])
X = np.array(Xread)
#print(X)
X = normalizeColumns(X)

K=2

findClusters(X,K,plot='i')



