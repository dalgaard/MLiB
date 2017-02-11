import numpy as np
import numpy.linalg as npl
from random import randrange,random
import matplotlib.pyplot as plt

class Hmm(object):
    
    __slots__ = ['hidden', 'observables', 'pi', 'A', 'emissions']
    
    def hiddenStates(self):
        '''
        
        '''
        return self.hidden

    def observableStates(self):
        '''
        
        '''
        return self.observables

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
            
