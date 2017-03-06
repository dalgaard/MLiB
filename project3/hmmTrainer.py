import sys
sys.path.append('../project2')
from random import random
from hmmTools import *
from hmmTestAgainstProject2 import *
from Data import Counts


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
        
    def train(self,observed,maxIt=2000,tol=1e-1):
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
                ll += sa.getDataLogLikelihood()
                
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
            print(self.hmm.pi)
            it += 1
            
class ViterbiTrainer(HmmTrainer):
    
    def __init__(self, hmm):
        HmmTrainer.__init__(self, hmm)
        
    def train(self,observed,pseudo=0.0,nTraces=1,maxIt=2000,tol=1e-4,piConstraints=[],aConstraints=[],phiConstraints=[]):
        K = self.hmm.K
        N = self.hmm.N
        
        diff = float('inf')
        it = 0
        prev = 0.0
        hid = self.hmm.hidden
        obs = self.hmm.observables
        
        print(self.header)
        
        while( diff > tol and it < maxIt ):
            newPi = [ pseudo for i in range(K) ]
            newA = [[ pseudo for i in range(K) ] for j in range(K)]
            newPhi = [[ pseudo for i in range(N)] for j in range(K)]
            ll = 0.0
            for iseq,seq in enumerate(observed):
                sa = ViterbiSequenceAnalyzer(self.hmm,seq,setB=1)
                ll += sa.getPosterior()
                for iTrace in range(nTraces):
                    c = Counts([seq],[sa.getTraceN(iTrace)])
                    for k,src in enumerate(hid):
                        newPi[k] += c.piCount.get(src, 0) 
                    
                    for i,src in enumerate(hid):
                        for j,dest in enumerate(hid):
                            newA[i][j] += c.transitionCount.get((src, dest), 0) 
                            
                    for i,src in enumerate(hid):
                        for j,dest in enumerate(obs):
                            newPhi[i][j] += c.emissionCount.get((src, dest), 0)
            
            
            diff = float("inf") if it == 0 else abs(ll-prev)
            self.currentState = "#"+"{:8d}".format(it)+"\t"+"{:10.6g}".format(ll)+"\t"+"{:.6g}".format(diff)
            print(self.currentState)
            if( diff<tol or it>maxIt):
                break
            
            prev = ll
            
            # impose the constraints
            for c in piConstraints:
                newPi[c] = pseudo
            for i,j in aConstraints:
                newA[i][j] = pseudo
            for i,j in phiConstraints:
                newPhi[i][j] = pseudo
            # update the parameters and calculate the new complete data log likelihood
            self.hmm.update(newPi,newA,newPhi)
            self.hmm.normalize()
            it += 1
