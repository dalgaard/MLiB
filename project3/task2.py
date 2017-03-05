import os,sys
sys.path.append('../project2')
from Data import Data, Counts
from hmmTools import Hmm
from hmmTestAgainstProject2 import *

M2o = 'O'
M2i = 'I'
NEW_HIDDEN = ['i', M2i, M2o, 'o']

def normalise(c):
    s = sum(c)
    return [x/s for x in c]


def learnAndPrint4StateModel(fileNames, printModel=False):
    data = Data.fromFiles(fileNames)
    data4states = []
    hidden = data.hiddenStates
    observables = data.observableStates
    for d in data:
        hid = d.hidden
        newHid = ""
        for h in hid:
            if h == 'i' or h == 'o':
                newHid += h
            else:
                newHid += M2o if newHid.endswith(M2o) or newHid.endswith('i') else M2i
        data4states.append(Data.Element(d.name, d.observed, newHid))
    data = Data(data4states, NEW_HIDDEN, observables)
    c = Counts.fromData(data)
    piC = [c.piCount.get(h, 0) for h in NEW_HIDDEN]
    pi = normalise(piC)
    A = [normalise([c.transitionCount.get((src, dest), 0) for dest in NEW_HIDDEN]) for src in NEW_HIDDEN]
    emissions = [normalise([c.emissionCount.get((h, o), 0) for o in observables]) for h in NEW_HIDDEN]
    hmm = Hmm(NEW_HIDDEN, observables, pi, A, emissions)
    if printModel:
        hmm.printRepr()
    
    testAgainstProj2(hmm)
    return(hmm)


if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(9)]
    learnAndPrint4StateModel(fileNames)

