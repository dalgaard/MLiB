import sys
sys.path.append('../project2')
from random import random
from hmmTools import *
from hmmTestAgainstProject2 import *


class HmmTrainer(object):
    __slots__ = ['hmm','header','currentState']
    def __init__(self, hmm):
        self.hmm = hmm
        self.hmm.normalize()
        self.header = '# Iteration\tlog p(X|Model)\tdifference'
        self.currentState = ''
        
    # define start parameters as random
    def randomInitialValues(self):
        K = self.hmm.K
        N = self.hmm.N
        self.hmm.update([ random() for i in range(K) ], [[ random() for i in range(K) ] for j in range(K)], [[ random() for i in range(N) ] for j in range(K)])
        
    def dump(self,fname="trained-hmm.txt"):
        myFile = open(fname,'w') 
        myFile.write("# This hmm has been trained by the em algorithm\n")
        myFile.write(self.header+"\n")
        myFile.write(self.currentState+"\n")
        self.hmm.dump(fname=myFile)
        myFile.close()
        
class PosteriorTrainer(HmmTrainer):
    
    __slots__ = ['sequenceAnalyzer']
    def __init__(self, sequenceAnalyzer, hmm):
        HmmTrainer.__init__(self, hmm)
        self.sequenceAnalyzer = sequenceAnalyzer
        
    def train(self,observed,maxIt=2000,tol=1e-4):
        K = self.hmm.K
        N = self.hmm.N
        
        diff = float('inf')
        it = 0
        prev = 0.0
        
        print(self.header)
        
        while( diff > tol and it < maxIt ):
            newPi = [ 0.0 for i in range(K) ]
            newA = [[ 0.0 for i in range(K) ] for j in range(K)]
            newPhi = [[ 0.0 for i in range(N)] for j in range(K)]
            ll = 0.0
            for seq in observed:
                sa = self.sequenceAnalyzer(self.hmm,seq,setB=1)
                ll += sa.getLogLikelihood()
                
                # get pi contribution for current sequence
                gamma = sa.getGamma(0)
                for k in range(K):
                    newPi[k] += gamma[k] 
                
                # get A contribution for current sequence
                for n in range(1,len(seq)):
                    zeta = sa.getZeta(n)
                    for kk in range(K):
                        for k in range(K):
                            newA[kk][k] += zeta[kk][k]
                
                # get the phi contribution
                for n in range(0,len(seq)):
                    emIdx = self.hmm.observables.index(seq[n])
                    gamma = sa.getGamma(n)
                    for kk in range(K):
                        newPhi[kk][emIdx] += gamma[kk]
            
            diff = float("inf") if it == 0 else ll-prev
            self.currentState = "#"+"{:8d}".format(it)+"\t"+"{:10.6g}".format(ll)+"\t"+"{:.6g}".format(diff)
            print(self.currentState)
            if( diff<tol or it>maxIt):
                break
            
            prev = ll
            
            # update the parameters and calculate the new complete data log likelihood
            self.hmm.update(newPi,newA,newPhi)
            self.hmm.normalize()
            it += 1
            
class ViterbiTrainer(HmmTrainer):
    
    __slots__ = ['sequenceAnalyzer']
    def __init__(self, sequenceAnalyzer, hmm):
        HmmTrainer.__init__(self, hmm)
        self.sequenceAnalyzer = sequenceAnalyzer
        
    def train(self,observed,maxIt=2000,tol=1e-4):
        K = self.hmm.K
        N = self.hmm.N
        
        diff = float('inf')
        it = 0
        prev = 0.0
        
        print(self.header)
        
        while( diff > tol and it < maxIt ):
            newPi = [ 0.0 for i in range(K) ]
            newA = [[ 0.0 for i in range(K) ] for j in range(K)]
            newPhi = [[ 0.0 for i in range(N)] for j in range(K)]
            ll = 0.0
            for seq in observed:
                sa = self.sequenceAnalyzer(self.hmm,seq,setB=1)
                ll += sa.getLogLikelihood()
                
                # get pi contribution for current sequence
                gamma = sa.getGamma(0)
                for k in range(K):
                    newPi[k] += gamma[k] 
                
                # get A contribution for current sequence
                for n in range(1,len(seq)):
                    zeta = sa.getZeta(n)
                    for kk in range(K):
                        for k in range(K):
                            newA[kk][k] += zeta[kk][k]
                
                # get the phi contribution
                for n in range(0,len(seq)):
                    emIdx = self.hmm.observables.index(seq[n])
                    gamma = sa.getGamma(n)
                    for kk in range(K):
                        newPhi[kk][emIdx] += gamma[kk]
            
            diff = float("inf") if it == 0 else ll-prev
            self.currentState = "#"+"{:8d}".format(it)+"\t"+"{:10.6g}".format(ll)+"\t"+"{:.6g}".format(diff)
            print(self.currentState)
            if( diff<tol or it>maxIt):
                break
            
            prev = ll
            
            # update the parameters and calculate the new complete data log likelihood
            self.hmm.update(newPi,newA,newPhi)
            self.hmm.normalize()
            it += 1
