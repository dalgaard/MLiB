import os, sys
sys.path.append('../project2')
from Data import Data, Counts
from hmmTools import Hmm

def normalise(c):
    s = sum(c)
    return [x/s for x in c]


def learnAndPrintModel(fileNames):
    d = Data.fromFiles(fileNames)
    hid = d.hiddenStates
    c = Counts.fromData(d)
    piC = [c.piCount.get(h, 0) for h in hid]
    pi = normalise(piC)
    A = [normalise([c.transitionCount.get((src, dest), 0) for dest in hid]) for src in hid]
    obs = d.observableStates
    emissions = [normalise([c.emissionCount.get((h, o), 0) for o in obs]) for h in hid]
    hmm = Hmm(hid, obs, pi, A, emissions)
    hmm.printRepr()


if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(9)]
    learnAndPrintModel(fileNames)
