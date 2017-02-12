from math import log

class Hmm(object):
    
    __slots__ = ['hidden', 'observables', 'pi', 'A', 'emissions']
    
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
