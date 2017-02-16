from math import log

class Hmm(object):
    
    __slots__ = ['hidden', 'observables', 'pi', 'A', 'emissions','K','N','alpha','beta','c']
    
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
        self.alpha = []
        self.beta = []
        self.c = []

    def logLikelihood(self, hiddenSequence, observedSequence):
        hidden_index = [ self.hidden.index(h) for h in hiddenSequence ]
        observed_index = [ self.observables.index(o) for o in observedSequence ]
        ll = 0.0
        for i in range(len(hidden_index)):
            if i == 0:
                ll += log(self.pi[hidden_index[0]])
            else:
                ll += log(self.A[hidden_index[i-1]][hidden_index[i]])
            ll += log(self.emissions[hidden_index[i]][observed_index[i]])
        return ll
    
    def forward(self):
        # initialize
        delta = [ self.pi[k] * self.emissions[k][0] for k in range(self.K) ]
        self.c = [ sum(delta) if n== 0 else 0.0 for n in range(self.N) ]
        self.alpha = [[ delta[k]/self.c[n] if n==0 else 0.0 for n in range(self.N) ] for k in range(self.K) ]
        
        # calculate subsequent steps
        for n in range(1,self.N):
            delta = [ 0.0 for k in range(self.K) ]
            for k in range(self.K):
                for kk in range(self.K):
                    delta[k] += self.alpha[kk][n-1] * self.A[k][kk]
                delta[k] = self.emissions[k][n] * delta[k]
            self.c[n] = sum(delta)
            for k in range(self.K):
                self.alpha[k][n] = delta[k] / self.c[n]
    
    def backward(self):
        # initialize
        self.beta = [[ 1.0 if n==self.N-1 else 0.0 for n in range(self.N) ] for k in range(self.K) ]
        
        # calculate subsequent steps
        for n in range(self.N-2,-1,-1):
            delta = [ 0.0 for k in range(self.K) ]
            for k in range(self.K):
                for kk in range(self.K):
                    delta[k] += self.beta[kk][n+1] * self.A[kk][k] * self.emissions[kk][n+1]
            for k in range(self.K):
                self.beta[k][n] = delta[k] / self.c[n+1]
        
            
