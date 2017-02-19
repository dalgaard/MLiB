from math import log
import numpy as np

class Hmm(object):
    
    __slots__ = ['hidden', 'observables', 'pi', 'A', 'emissions','K','N']
    
    def __init__(self, file):
        self.hidden = []
        self.observables = []
        with open(file,'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        i=0
        while i < len(content):
            if content[i] == 'hidden':
                self.hidden = content[i+1].split(' ')
                i+=2
            elif content[i] == 'observables':
                self.observables = content[i+1].split(' ')
                i+=2
            elif content[i] == 'pi':
                self.pi = [ float(x) for x in content[i+1].split(' ')]
                i+=2
            elif content[i] == 'transitions':
                self.A = [ [ float(y) for y in x.split(' ')] for x in content[i+1:i+1+len(self.hidden)] ]
                i += 2+len(self.hidden)
            elif content[i] == 'emissions':
                self.emissions = [ [ float(y) for y in x.split(' ')] for x in content[i+1:i+1+len(self.hidden)] ]
                i += 2+len(self.hidden)
            else:
                i += 1
        self.K = len(self.pi)
        self.N = len(self.emissions[0])
    
class HmmSequenceAnalyzer(object):
    __slots__ = ['Hmm','sequence', 'alpha','beta','c','viterbiTrace']
    
    def __init__(self, Hmm, observedSequence):
        self.Hmm = Hmm
        self.sequence = observedSequence
        self.forward()
        self.backward()
    
    def logLikelihood(self, hiddenSequence):
        hidden_index = [ self.Hmm.hidden.index(h) for h in hiddenSequence ]
        observed_index = [ self.Hmm.observables.index(o) for o in self.sequence ]
        ll = 0.0
        for i in range(len(hidden_index)):
            if i == 0:
                ll += log(self.Hmm.pi[hidden_index[0]])
            else:
                prob = self.Hmm.A[hidden_index[i-1]][hidden_index[i]]
                if abs(prob)<1e-13 :
                    return -float("Inf")
                ll += log(self.Hmm.A[hidden_index[i-1]][hidden_index[i]])
            ll += log(self.Hmm.emissions[hidden_index[i]][observed_index[i]])
        return ll
        
    
    def forward(self):
        N = len(self.sequence)
        # initialize
        delta = [ self.Hmm.pi[2-k] * self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])] for k in range(self.Hmm.K) ]
        self.c = [ sum(delta) if n== 0 else 0.0 for n in range(N) ]
        self.alpha = [[ delta[k]/self.c[n] if n==0 else 0.0 for n in range(N) ] for k in range(self.Hmm.K) ]
        omega = [ [ d if k == 0 else 0.0 for d in delta ] for k  in range(2) ]
        self.viterbiTrace = [ np.argmax(delta) if n==0 else -1 for n in range(N) ]
        
        # calculate subsequent steps
        for n in range(1,N):
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            delta = [ 0.0 for k in range(self.Hmm.K) ]
            for k in range(self.Hmm.K):
                maxP = 0.0
                kmax = 0
                for kk in range(self.Hmm.K):
                    delta[k] += self.alpha[kk][n-1] * self.Hmm.A[k][kk]
                    if maxP < omega[(n-1)%2][kk] * self.Hmm.A[k][kk] :
                        maxP = omega[(n-1)%2][kk] * self.Hmm.A[k][kk]
                        kmax = kk
                omega[n%2][k] = self.Hmm.emissions[k][emissionIDX] * maxP
                delta[k] = self.Hmm.emissions[k][emissionIDX] * delta[k]
            self.viterbiTrace[n] = np.argmax(omega[n%2])
            self.c[n] = sum(delta)
            for k in range(self.Hmm.K):
                self.alpha[k][n] = delta[k] / self.c[n]
    
    def backward(self):
        N = len(self.sequence)
        # initialize
        self.beta = [[ 1.0 if n==N-1 else 0.0 for n in range(N) ] for k in range(self.Hmm.K) ]
        
        # calculate subsequent steps
        for n in range(N-2,-1,-1):
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            delta = [ 0.0 for k in range(self.Hmm.K) ]
            for k in range(self.Hmm.K):
                for kk in range(self.Hmm.K):
                    delta[k] += self.beta[kk][n+1] * self.Hmm.A[kk][k] * self.Hmm.emissions[kk][emissionIDX]
            for k in range(self.Hmm.K):
                self.beta[k][n] = delta[k] / self.c[n+1]
        
    def getTrace(self,choice="viterbi"):
        trace = ""
        if(choice=="viterbi"):
            for n in range(len(self.viterbiTrace)):
                trace += self.Hmm.hidden[self.viterbiTrace[n]]
        else:
            for n in range(len(self.viterbiTrace)):
                trace += self.Hmm.hidden[self.getArgMaxPosterior(n)]
        return trace
        
    
    def printViterbiTrace(self,compact=True):
        if(len(self.viterbiTrace) == 0):
            self.forward()
        if(compact):
            for n in range(len(self.viterbiTrace)):
                print(self.Hmm.hidden[self.viterbiTrace[n]],end="")
            print()
        else:
            for n in range(len(self.viterbiTrace)):
                print("  "+self.Hmm.hidden[self.viterbiTrace[n]]+"  ",end="\t")
            print()
            for n in range(len(self.viterbiTrace)):
                print("{:3d} ".format(self.viterbiTrace[n]),end='\t')
            print()
            if( len(self.beta) != 0):
                for n in range(len(self.viterbiTrace)):
                    print("{:5.2f}".format(100*self.getPosterior(self.viterbiTrace[n],n)),end='\t')
                print()
    
            
    def getPosterior(self,k,n):
        if len(self.alpha) == 0 :
            self.Hmm.forward()
        if len(self.beta) == 0 :
            self.Hmm.backward()
        return self.alpha[k][n] * self.beta[k][n]

    def getArgMaxPosterior(self,n):
        maxP = 0.0
        kmax = 0
        for k in range(self.Hmm.K):
            if maxP < self.getPosterior(k,n) :
                maxP = self.getPosterior(k,n)
                kmax = k
        return kmax
