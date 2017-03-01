from logsumTools import *
import numpy as np

class Hmm(object):
    
    __slots__ = ['hidden', 'observables', 'pi', 'A', 'emissions','K','N']
    
    def __init__(self,hidden,observables,pi,A,emissions):
        self.hidden = hidden
        self.observables = observables
        
        self.K = len(self.hidden)
        self.N = len(self.observables)
        
        self.update(pi,A,emissions)
    
    @classmethod
    def fromFile(hmm,file):
        with open(file,'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        i=0
        while i < len(content):
            if content[i] == 'hidden':
                hidden = content[i+1].split(' ')
                i+=2
            elif content[i] == 'observables':
                observables = content[i+1].split(' ')
                i+=2
            elif content[i] == 'pi':
                pi = [ float(x) for x in content[i+1].split(' ')]
                i+=2
            elif content[i] == 'transitions':
                A = [ [ float(y) for y in x.split(' ')] for x in content[i+1:i+1+len(hidden)] ]
                i += 2+len(hidden)
            elif content[i] == 'emissions':
                emissions = [ [ float(y) for y in x.split(' ')] for x in content[i+1:i+1+len(hidden)] ]
                i += 2+len(hidden)
            else:
                i += 1
        return(hmm(hidden,observables,pi,A,emissions))
    
    def update(self,pi,A,emissions):
        if( len(pi) != self.K):
            print("error in update, wrong size of pi")
        self.pi = pi
        if( len(A) != self.K or len(A[0])!=self.K):
            print("error in update, wrong size of A")
        self.A = A
        if( len(emissions) != self.K or len(emissions[0])!=self.N):
            print("error in update, wrong size of emissions")
        self.emissions = emissions
    
class HmmSequenceAnalyzer(object):
    
    __slots__ = ['Hmm','sequence','B']
    
    def __init__(self, Hmm, observedSequence,setB=0):
        self.Hmm = Hmm
        self.sequence = observedSequence
        
        # choose default B as the square root of the number of 
        if( setB > 0 ):
            self.B = setB
        else:
            self.B = math.floor(len(self.sequence)**(0.5)) 
    
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
    
    def getConstants(self):
        # constants for bookkeeping
        N = len(self.sequence)
        K = self.Hmm.K
        B = self.B
        S = math.ceil(N/B)
        return N,K,B,S
        
    def getTrace(self):
        return ""
    

class ViterbiSequenceAnalyzer(HmmSequenceAnalyzer):
    
    __slots__ = ['omega','viterbiTrace','work','lastCol']
    
    def __init__(self, Hmm, observedSequence, setB=0):
        HmmSequenceAnalyzer.__init__(self,Hmm,observedSequence,setB=setB)
        self.work=[[float("-inf") for k in range(Hmm.K)] for n in range(2)]
        self.forward()
    
    def forward(self):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        delta = [ log(self.Hmm.pi[k]) + log(self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])]) for k in range(K)]

        # initialize omega and do the forward sweep storing a bunch of columns
        self.omega = [[ delta[k] if n==0 else float("-inf") for k  in range(K)] for n in range(S)]
        self.lastCol = self.loopForward(delta,0,N-1,True)
        
        # initialize
        self.viterbiTrace = [ np.argmax(self.lastCol) if n==N-1 else -1 for n in range(N) ]
        
        # calculate subsequent steps
        for n in range(N-2,-1,-1):
            kprev = self.viterbiTrace[n+1]
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            ompri = [ -float("Inf") for k in range(K) ]
            sO = n//B
            IO = sO*B
            omega = self.loopForward(self.omega[sO],IO,n,False)
            for k in range(K):
                ompri[k] = omega[k] + log(self.Hmm.A[k][kprev]) + log(self.Hmm.emissions[kprev][emissionIDX]) 
            self.viterbiTrace[n] = np.argmax(ompri)

    # start and end inclusive, start is the idx of the firstCol and end is the index of the column to be returned
    def loopForward(self, firstCol ,start,end,fillOmega):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        #initialize work
        for k in range(K):
            self.work[0][k]=firstCol[k]
        
        # calculate subsequent steps
        nidx = 0
        for n in range(start+1,end+1):
            nidx+=1
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            for k in range(K):
                T = float("-inf")
                for kk in range(K):
                    t = self.work[(nidx-1)%2][kk] + log(self.Hmm.A[kk][k])
                    if t>T:
                        T=t
                self.work[nidx%2][k] = log(self.Hmm.emissions[k][emissionIDX]) + T
                if(n%B==0 and fillOmega):
                    self.omega[n//B][k] = self.work[nidx%2][k]
                    
        return [self.work[nidx%2][k] for k in range(K)]
        
    def getTrace(self):
        trace = ""
        for n in range(len(self.sequence)):
            trace += self.Hmm.hidden[self.viterbiTrace[n]]
        return trace

    def getPosterior(self):
        return max(self.lastCol)


class ViterbiHatSequenceAnalyzer(HmmSequenceAnalyzer):
    
    __slots__ = ['omegaHat']
    
    def __init__(self, Hmm, observedSequence):
        HmmSequenceAnalyzer.__init__(self,Hmm,observedSequence)
        self.viterbiHat()

    def viterbiHat(self):
        x = self.sequence
        obs = self.Hmm.observables
        emis = np.log(self.Hmm.emissions)
        pi = np.log(self.Hmm.pi)
        A = np.log(self.Hmm.A)
        xIdx = [ obs.index(y) for y in x ]
        omegaHat = [ [pi[k] + emis[k][xIdx[0]] if n == 0 else -float('Inf') for n in range(len(x))] for k in range(len(pi)) ]
        for n in range(1, len(x)):
            for k in range(len(pi)):
                tmp = [ emis[k][xIdx[n]] + omegaHat[j][n-1] + A[j][k] for j in range(len(pi)) ]
                omegaHat[k][n] = np.max(tmp)
        self.omegaHat = omegaHat

    def viterbiHatTrace(self):
        x = self.sequence
        hid = self.Hmm.hidden
        emis = np.log(self.Hmm.emissions)
        pi = np.log(self.Hmm.pi)
        A = np.log(self.Hmm.A)
        xIdx = [ self.Hmm.observables.index(y) for y in x ]
        maxOmegaN = np.argmax([self.omegaHat[k][len(x) - 1] for k in range(len(hid))])
        zIdx = [ maxOmegaN if i == len(x) - 1 else -1 for i in range(len(x))]
        for n in reversed(range(len(x) - 1)):
            z1 = zIdx[n+1]
            x1 = xIdx[n+1]
            emis1 = emis[z1][x1]
            tmp = [ emis1 + self.omegaHat[k][n] + A[k][z1] for k in range(len(hid)) ]
            zIdx[n] = np.argmax(tmp)
        return ''.join([ hid[z] for z in zIdx ])

    def getTrace(self):
        return self.viterbiHatTrace()

    def getPosterior(self):
        lastCol = [ self.omegaHat[k][len(self.omegaHat[0]) -1] for k in range(len(self.omegaHat)) ]
        return max(lastCol)

class ScaledPosteriorSequenceAnalyzer(HmmSequenceAnalyzer):
    
    __slots__ = ['alpha','beta','c']
    
    def __init__(self, Hmm, observedSequence, setB=0):
        HmmSequenceAnalyzer.__init__(self,Hmm,observedSequence,setB=setB)
        self.forward()
        self.backward()
        
    def forward(self):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)

        # delta contains the initial probabilities for the forward and viterbi
        delta = [ self.Hmm.pi[k] * self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])] for k in range(K) ]
        
        # initialize c as the sum of the deltas
        self.c = [ sum(delta) if n==0 else 0.0 for n in range(N) ]
        
        # initialize alpha and omega
        self.alpha = [[ delta[k]/self.c[n] if n==0 else 0.0 for n in range(N) ] for k in range(K) ]
        
        # calculate subsequent steps
        for n in range(1,N):
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            delta = [ 0.0 for k in range(K) ]
            for k in range(K):
                for kk in range(K):
                    delta[k] += self.alpha[kk][n-1] * self.Hmm.A[kk][k]
                delta[k] = self.Hmm.emissions[k][emissionIDX] * delta[k]
            self.c[n] = sum(delta)
            for k in range(K):
                self.alpha[k][n] = delta[k] / self.c[n]
    
    def backward(self):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        # initialize
        self.beta = [[ 1.0 if n==N-1 else 0.0 for n in range(N) ] for k in range(K) ]
        
        # calculate subsequent steps
        for n in range(N-2,-1,-1):
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            for k in range(K):
                for kk in range(K):
                    # here we could have pulled the scaling out, for simpler code structure it is kept here
                    self.beta[k][n] += (self.beta[kk][n+1]/self.c[n+1]) * self.Hmm.A[k][kk] * self.Hmm.emissions[kk][emissionIDX]
               
        
    def getTrace(self):
        trace = ""
        for n in range(len(self.sequence)):
            trace += self.Hmm.hidden[self.getArgMaxPosterior(n)]
        return trace
    
    def getPosterior(self,k,n):
        return self.alpha[k][n] * self.beta[k][n]
    
    def getArgMaxPosterior(self,n):
        return np.argmax([self.getPosterior(k,n) for k in range(self.Hmm.K)])
    

class LogSumSequenceAnalyzer(HmmSequenceAnalyzer):
    
    __slots__ = ['alpha','beta','work']
    
    def __init__(self, Hmm, observedSequence, setB=0):
        HmmSequenceAnalyzer.__init__(self,Hmm,observedSequence, setB=setB)
        self.work=[[float("-inf")  for k in range(Hmm.K)] for n in range(2)]
        self.forward()
        self.backward()
    
    def forward(self):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        delta = [ log(self.Hmm.pi[k]) + log(self.Hmm.emissions[k][self.Hmm.observables.index(self.sequence[0])]) for k in range(K)]
        
        # initialize alpha and run the forward algorithm with checkpointing
        self.alpha = [[ delta[k] if n==0 else float("-inf")  for k in range(K)] for n in range(S) ]
        self.loopForward(delta,0,N-1,True)
        
    # start and end inclusive, start is the idx of the firstCol and end is the index of the column to be returned
    def loopForward(self, firstCol ,start,end,fillAlpha):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        #initialize work matrix
        for k in range(K):
            self.work[0][k] = firstCol[k]
        
        # calculate subsequent steps
        nidx = 0
        for n in range(start+1,end+1):
            nidx+=1
            emissionIDX = self.Hmm.observables.index(self.sequence[n])
            for k in range(K):
                ls = float("-inf")
                for kk in range(K):
                    ls = logsum(ls, self.work[(nidx-1)%2][kk] + log(self.Hmm.A[kk][k]))
                self.work[nidx%2][k] = ls+log(self.Hmm.emissions[k][emissionIDX])
                if(n%B==0 and fillAlpha):
                    for k in range(K):
                        self.alpha[n//B][k] = self.work[nidx%2][k]
                    
        return [self.work[nidx%2][k] for k in range(K)]
    
    def backward(self):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        # initialize
        self.beta = [[ 0.0 if n==S-1 else float("-inf") for k in range(K) ] for n in range(S) ]
        self.loopBackward([0.0 for k in range(K)],N-1,0,True)
        
    # start and end inclusive, start is the idx of the firstCol and end is the index of the column to be returned
    def loopBackward(self,firstCol,start,end,fillBeta):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        #initialize work matrix
        for k in range(K):
            self.work[0][k] = firstCol[k]
            self.work[1][k] = float("-inf")
        
        # calculate subsequent steps
        nidx = 0
        for n in range(start-1,end-1,-1):
            nidx += 1
            emissionIDX = self.Hmm.observables.index(self.sequence[n+1])
            for k in range(K):
                for kk in range(K):
                    self.work[nidx%2][k] = logsum(self.work[nidx%2][k],self.work[(nidx+1)%2][kk] + log(self.Hmm.A[k][kk]) + log(self.Hmm.emissions[kk][emissionIDX]))
                    
            # reinit work for next round
            for k in range(K):
                self.work[(nidx+1)%2][k] = float("-inf")
            
            # checkpoint the beta matrix
            if((N-1-n)%B==0 and fillBeta):
                for k in range(K):
                    self.beta[n//B][k] = self.work[nidx%2][k]
                    
        return [self.work[nidx%2][k] for k in range(K) ]
    
        
    def getTrace(self):
        trace = ""
        for n in range(len(self.sequence)):
            trace += self.Hmm.hidden[self.getArgMaxPosterior(n)]
        return trace
    
    def getArgMaxPosterior(self,n):
        N,K,B,S = HmmSequenceAnalyzer.getConstants(self)
        
        # alpha starting indices
        sA = n//B
        IA = sA*B
        #print(sA,IA,len(self.alpha),S,B,self.B,N,math.ceil(N/B))
        a = self.loopForward([self.alpha[sA][k] for k in range(K)],IA,n,False)
        
        # beta starting indices
        sP = (N-1-n)//B
        sB = S-1-sP
        IB = N-1-sP*B
        b = self.loopBackward([self.beta[sB][k] for k in range(K)],IB,n,False)
        
        return np.argmax([ a[k] + b[k] for k in range(self.Hmm.K)])
