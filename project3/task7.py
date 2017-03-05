import os,sys
sys.path.append('../project2')
from Data import *
from random import random
from hmmTools import *
from hmmTestAgainstProject2 import *
from hmmTrainer import ViterbiTrainer
from startingGuesses import getUnnormalizedStartingGuess3State, getUnnormalizedStartingGuess4State


def learnAndPrintModel(fileNames, model):
    d = Data.fromFiles(fileNames)
    
    N = len(d.observableStates)
    if( model == '3State'):
        hidden = ['i','M','o']
        piStart, AStart, phiStart = getUnnormalizedStartingGuess3State(random,len(hidden),N) #random starting guess
        #piStart, AStart, phiStart = getStartingGuess(lambda : 1.0) #uniform starting guess
    elif( model == '4State'):
        hidden = ['i','I','O','o'] # read as (i)nside, (I)nwards, (O)utwards, (o)utside
        piStart, AStart, phiStart = getUnnormalizedStartingGuess4State(random,len(hidden),N) #random starting guess
        
    # This should also work with LogSumSequenceAnalyzer
    hT = ViterbiTrainer(Hmm(hidden,d.observableStates,piStart,AStart,phiStart))
    #hT = PosteriorTrainer(LogSumSequenceAnalyzer, Hmm(hidden,d.observableStates,piStart,AStart,phiStart))
        
    hT.dump('initial-parameters-'+model+'-viterbi.txt')
    hT.train(d.getObserved(),pseudo=1.0,tol=1e-4,maxIt=1000)
    hT.dump('final-parameters-'+model+'-viterbi.txt')
    hT.hmm.printRepr()

if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(9)]
    learnAndPrintModel(fileNames, '3State')
    learnAndPrintModel(fileNames, '4State')
