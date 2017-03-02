import sys
sys.path.append('../project2')
from random import random
from hmmTools import *

def getAnalysis(f,sequenceAnalyzer):
    trace=sequenceAnalyzer.getTrace()
    return "# "+trace+"\n; log P(x,z) = "+str(sequenceAnalyzer.logLikelihood(trace))+"\n\n"

class HmmTrainer(object):
    __slots__ = ['hmm']
    def __init__(self, hmm):
        self.hmm = hmm
        
    # define start parameters as random
    def randomInitialValues(self):
        K = self.hmm.K
        N = self.hmm.N
        self.hmm.update([ random() for i in range(K) ], [[ random() for i in range(K) ] for j in range(K)], [[ random() for i in range(N) ] for j in range(K)])
        
    def train(self,observed,maxIt=2000,tol=1e-4):
        K = self.hmm.K
        N = self.hmm.N
        
        diff = float('inf')
        it = 0
        prev = 0.0
        
        while( diff > tol and it < maxIt ):
            newPi = [ 0.0 for i in range(K) ]
            newPiDenom = 0.0
            newA = [[ 0.0 for i in range(K) ] for j in range(K)]
            newADenom = [ 0.0 for i in range(K) ]
            newPhi = [[ 0.0 for i in range(N)] for j in range(K)]
            newPhiDenom = [0.0 for i in range(K)]
            ll = 0.0
            for seq in observed:
                sa = ScaledPosteriorSequenceAnalyzer(self.hmm,seq)
                
                # get pi contribution for current sequence
                gamma = sa.getGamma(0)
                for k in range(K):
                    newPi[k] += gamma[k] 
                    newPiDenom += gamma[k]
                    if( self.hmm.pi[k] > 1e-13):
                        ll += gamma[k] * log(self.hmm.pi[k])
                
                # get A contribution for current sequence
                for n in range(1,len(seq)):
                    zeta = sa.getZeta(n)
                    for kk in range(K):
                        for k in range(K):
                            newA[kk][k] += zeta[kk][k]
                            newADenom[kk] += zeta[kk][k]
                            if self.hmm.A[kk][k] > 1e-13 :
                                ll += zeta[kk][k] * log(self.hmm.A[kk][k])
                
                # get the phi contribution
                for n in range(0,len(seq)):
                    emIdx = self.hmm.observables.index(seq[n])
                    gamma = sa.getGamma(n)
                    for kk in range(K):
                        newPhi[kk][emIdx] += gamma[kk]
                        newPhiDenom[kk] += gamma[kk]
                        if( self.hmm.emissions[kk][emIdx] > 1e-13 ):
                            ll += gamma[kk] * log(self.hmm.emissions[kk][emIdx])
            
            diff = abs(prev-ll)
            print("iteration",it,"total",ll,"diff",diff)
            if( diff<tol or it>maxIt):
                break
            
            prev = ll
            
            # normalize pi, A, phi
            for k in range(K):
                newPi[k] = newPi[k]/newPiDenom
                for kk in range(K):
                    newA[k][kk]  = newA[k][kk]/newADenom[k]
                for n in range(N):
                    newPhi[k][n] = newPhi[k][n]/newPhiDenom[k]
            
            
            # update the parameters and calculate the new complete data log likelihood
            self.hmm.update(newPi,newA,newPhi)
            it += 1
            

observed=[]
with open('./Dataset160/set160.0.labels.txt','r') as f:
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.strip().startswith(">") :
            observed.append(lines[iline+1].strip())
hidden = ['i','M','o']
observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
K = len(hidden)
N = len(observables)
piStart  =  [ random() for i in range(K) ]
AStart   = [[ random() for i in range(K) ] for j in range(K)]
#set the io and oi transition probabilities to 0
AStart[0][K-1] = 0.0 
AStart[K-1][0] = 0.0
phiStart = [[ random() for i in range(N) ] for j in range(K)]
hT = HmmTrainer(Hmm(hidden,observables,piStart,AStart,phiStart))
hT.train(observed)


with open('../project2/test-sequences-project2.txt','r') as f:
    post = open("posterior-project2.txt",'w')
    post.write("Posterior-decoding using the scaled forward and backward algorithms\n")
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.startswith(">"):
            for f in [post]:
                f.write(line.strip()+"\n")
                f.write(lines[iline+1].strip()+"\n")
            # the scaled posterior and the viterbi decodings are calculated at the same time
            post.write(getAnalysis(post,ScaledPosteriorSequenceAnalyzer(hT.hmm,lines[iline+1].strip())))
            post.write(getAnalysis(post,ViterbiSequenceAnalyzer(hT.hmm,lines[iline+1].strip())))
