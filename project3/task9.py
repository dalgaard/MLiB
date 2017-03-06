import os,sys
sys.path.append('../project2')
import time
from Data import *
from random import random
from hmmTools import *
from hmmTestAgainstProject2 import *
from hmmTrainer import PosteriorTrainer


def learnAndPrintModel(fileNames, sa):
    d = Data.fromFiles(fileNames)
    
    N = len(d.observableStates)
    hidden = ['i','I','O','o'] # read as (i)nside, (I)nwards, (O)utwards, (o)utside
    hT = PosteriorTrainer(sa, Hmm.fromFile('./trained4StateViterbiRandom/final-parameters-4State-viterbi.txt'))
    hT.train(d.getObserved(),tol=1e-1,maxIt=1000)
    hT.hmm.printRepr()

if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(9)]
    t1 = time.process_time()
    learnAndPrintModel(fileNames, ScaledPosteriorSequenceAnalyzer)
    t2 = time.process_time()
    learnAndPrintModel(fileNames, LogSumSequenceAnalyzer)
    t3 = time.process_time()
    print("Time Scaled:",t2-t1,"Time Logsum:",t3-t2)
