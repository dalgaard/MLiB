import sys
sys.path.append('../project2')
from random import random
from hmmTools import *

class HmmTrainer(object):
    __slots__ = ['hmm','piStart','AStart','phiStart','gamma','zeta']
    def __init__(self, hmm):
        self.hmm = hmm
        
    # define start parameters as random
    def randomInitialValues(self):
        O = self.hmm.observables
        H = self.hmm.hidden
        K = len(H)
        
        self.piStart  =  [ random() for i in range(K) ]
        self.AStart   = [[ random() for i in range(K) ] for j in range(K)]
        self.phiStart = [[ random() for i in range(K) ] for j in range(len(O))]
    # define start parameters from input
    def setInitialValues(self,pi,A,phi):
        self.piStart  = pi 
        self.AStart   = A
        self.phiStart = phi
        
    def train(self,observed,maxIt=20,tol=1e-4):
        
        diff = float('inf')
        it = 0
        
        while( diff > tol and it < maxIt ):
            for seq in observed:
                exit()

hT = HmmTrainer(Hmm.fromFile('../project2/hmm-tm.txt'))
hT.randomInitialValues()
