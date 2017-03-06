import os,sys
sys.path.append('../project2')
from Data import *
from random import random
from hmmTools import *
from hmmTestAgainstProject2 import *
from hmmTrainer import PosteriorTrainer
from startingGuesses import *


def learnAndPrintModel(fileNames, model):
    d = Data.fromFiles(fileNames)
    
    N = len(d.observableStates)
    hidden = ['i','I','O','p','q'] 
    cA = getAConstraints(hidden,["ip","iq","pi","qi","IO","OI","iI","pO","qO","Oi","Ip","Iq"])
    cP = getPiConstraints(hidden,["I","O"])
                    
    fun = lambda : 1.0      #uniform starting guess
    piStart, AStart, phiStart = getUnnormalizedStartingGuess(fun,len(hidden),N,piConstraints=cP, aConstraints=cA)
        
    # This should also work with LogSumSequenceAnalyzer
    sa = ScaledPosteriorSequenceAnalyzer
    #sa = LogSumSequenceAnalyzer
    hT = PosteriorTrainer(sa, Hmm(hidden,d.observableStates,piStart,AStart,phiStart))
        
    hT.dump('initial-parameters-'+model+'.txt')
    hT.train(d.getObserved(),tol=1e-4,maxIt=1000)
    hT.dump('final-parameters-'+model+'.txt')
    hT.hmm.printRepr()

if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(9)]
    learnAndPrintModel(fileNames, '5State')
