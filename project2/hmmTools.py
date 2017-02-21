from logsumTools import *
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
    
    __slots__ = ['Hmm','sequence', 'omega', 'alpha','beta','c','viterbiTrace']
    
    def __init__(self, Hmm, observedSequence, logVersion = False):
        self.Hmm = Hmm
        self.sequence = observedSequence
        if( logVersion ):
            self.logForward()
            self.logBackward()
        else:
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
        K = self.Hmm.K

        # delta contains the initial probabilities for the forward and viterbi
        delta = [ self.Hmm.pi[k] * self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])] for k in range(K) ]
        
        # initialize c as the sum of the deltas
        self.c = [ sum(delta) if n==0 else 0.0 for n in range(N) ]
        
        # initialize alpha and omega
        self.alpha = [[ delta[k]/self.c[n] if n==0 else 0.0 for n in range(N) ] for k in range(K) ]
        self.omega = [[ log(delta[k]) if n==0 else float("-inf") for n in range(N) ] for k  in range(K) ]
        
        # calculate subsequent steps
        for n in range(1,N):
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            delta = [ 0.0 for k in range(K) ]
            for k in range(K):
                ompri = [ float("-inf") for kk in range(K) ]
                for kk in range(K):
                    delta[k] += self.alpha[kk][n-1] * self.Hmm.A[kk][k]
                    if ( abs(self.Hmm.A[kk][k]) > 1e-13):
                        ompri[kk] = self.omega[kk][n-1] + log(self.Hmm.A[kk][k])
                self.omega[k][n] = log(self.Hmm.emissions[k][emissionIDX]) + max(ompri)
                delta[k] = self.Hmm.emissions[k][emissionIDX] * delta[k]
            self.c[n] = sum(delta)
            for k in range(K):
                self.alpha[k][n] = delta[k] / self.c[n]
    
    def logForward(self):
        N = len(self.sequence)
        K = self.Hmm.K
        sqrtN = math.floor(N**(0.5)) 
        slots = math.ceil(N/sqrtN)
        delta = [ self.Hmm.pi[k] * self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])] for k in range(K)]
        
        # initialize alpha and omega
        self.alpha = [[ log(delta[k]) if n==0 else float("-inf") for n in range(slots) ] for k in range(K) ]
        self.loopForward([log(d) for d in delta],1,N-1,True)
        
    def loopForward(self, firstCol ,start,end,fillAlpha):
        N = len(self.sequence)
        K = self.Hmm.K
        sqrtN = math.floor(N**(0.5)) 
        slots = math.ceil(N/sqrtN)
        work=[[ firstCol[k] if n==0 else float("-inf") for n in range(2) ] for k in range(K) ]
        
        # calculate subsequent steps
        nidx = 0
        for n in range(start,end+1):
            nidx+=1
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            for k in range(K):
                ls = float("-inf")
                for kk in range(K):
                    ls = logsum(ls, work[kk][(nidx-1)%2] + log(self.Hmm.A[kk][k]))
                work[k][nidx%2] = ls+log(self.Hmm.emissions[k][emissionIDX])
                if(n%sqrtN==0 and fillAlpha):
                    print(n,N,sqrtN,n//sqrtN,N//sqrtN,slots,N)
                    self.alpha[k][n//sqrtN] = work[k][nidx%2]
        return [work[k][nidx%2] for k in range(K)]
    
    def backward(self):
        N = len(self.sequence)
        K = self.Hmm.K
        
        # initialize
        self.beta = [[ 1.0 if n==N-1 else 0.0 for n in range(N) ] for k in range(K) ]
        self.viterbiTrace = [ np.argmax([self.omega[k][n] for k in range(K)]) if n==N-1 else -1 for n in range(N) ]
        
        # calculate subsequent steps
        for n in range(N-2,-1,-1):
            kprev = self.viterbiTrace[n+1]
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            ompri = [ -float("Inf") for k in range(K) ]
            for k in range(K):
                for kk in range(K):
                    self.beta[k][n] += (self.beta[kk][n+1]/self.c[n+1]) * self.Hmm.A[k][kk] * self.Hmm.emissions[kk][emissionIDX]
                if( self.Hmm.emissions[kprev][emissionIDX] > 1e-13 and  self.Hmm.A[k][kprev] > 1e-13 ):
                    ompri[k] = self.omega[k][n] + log(self.Hmm.A[k][kprev]) + log(self.Hmm.emissions[kprev][emissionIDX]) 
            self.viterbiTrace[n] = np.argmax(ompri)
               
    def logBackward(self):
        N = len(self.sequence)
        K = self.Hmm.K
        sqrtN = math.floor(N**(0.5)) 
        slots = math.ceil(N/sqrtN)
        # initialize
        self.beta = [[ 0.0 if n==N-1 else float("-inf") for n in range(slots) ] for k in range(K) ]
        self.loopBackward([0.0 for k in range(K)],N-2,0,True)
        
    def loopBackward(self,firstCol,start,end,fillBeta):
        N = len(self.sequence)
        K = self.Hmm.K
        sqrtN = math.floor(N**(0.5)) 
        slots = math.ceil(N/sqrtN)
        work=[[firstCol[k] if n==0 else float("-inf") for n in range(2) ] for k in range(K) ]
        # calculate subsequent steps
        nidx = 0
        for n in range(start,end-1,-1):
            nidx += 1
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            for k in range(K):
                for kk in range(K):
                    work[k][nidx%2] = logsum(work[k][nidx%2],work[kk][(nidx+1)%2] + log(self.Hmm.A[k][kk]) + log(self.Hmm.emissions[kk][emissionIDX]))
            if((N-1-n)%sqrtN==0 and fillBeta):
                for k in range(K):
                    self.beta[k][n//sqrtN] = work[k][nidx%2]
        return [work[k][nidx%2] for k in range(K)]
        
    def getTrace(self,choice="viterbi"):
        trace = ""
        if(choice=="viterbi"):
            for n in range(len(self.sequence)):
                trace += self.Hmm.hidden[self.viterbiTrace[n]]
        elif choice == "posterior":
            for n in range(len(self.sequence)):
                trace += self.Hmm.hidden[self.getArgMaxPosterior(n)]
        elif choice == "logPosterior":
            for n in range(len(self.sequence)):
                trace += self.Hmm.hidden[self.getArgMaxLogPosterior(n)]
        else:
            print("unrecognized option: "+choice)
            exit()
        return trace
    
    def getPosterior(self,k,n):
        if len(self.alpha) == 0 :
            self.Hmm.forward()
        if len(self.beta) == 0 :
            self.Hmm.backward()
        return self.alpha[k][n] * self.beta[k][n]
    
    def getArgMaxPosterior(self,n):
        return np.argmax([self.getPosterior(k,n) for k in range(self.Hmm.K)])
    
    def getArgMaxLogPosterior(self,requestedN):
        K = self.Hmm.K
        N = len(self.sequence)
        sqrtN = math.floor(N**(0.5)) 
        slots = math.ceil(N/sqrtN)
        # get the right alpha
        nidx = requestedN//sqrtN
        startIdx = nidx*sqrtN
        a = self.loopForward([self.alpha[k][nidx] for k in range(K)] ,startIdx+1,requestedN,False)
        
        # beta starting index
        nidx = math.floor((N-1-requestedN)/sqrtN)
        startIdx = nidx*sqrtN
        print(requestedN,nidx,startIdx,N,sqrtN)
        b = self.loopBackward([self.beta[k][nidx] for k in range(K)], min(startIdx-1,N-1), requestedN,False)
        return np.argmax([ a[k] + b[k] for k in range(self.Hmm.K)])
