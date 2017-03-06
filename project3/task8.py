import os,sys
sys.path.append('../project2')
from Data import *
from random import random
from hmmTools import *
from hmmTrainer import *
from hmmTestAgainstProject2 import *
from hmmTrainer import ViterbiTrainer
from startingGuesses import *
from subprocess import Popen, PIPE
from numpy import nanmean, nanstd, nanvar
import re

def predict(hmm, file, outFile):
    """
    :param hmm:the hmm
    :type hmm: Hmm
    :param file: input file
    :type file: str
    :param outFile: output file
    :type outFile; str
    """
    trueOutputName = "{}_true.txt".format(outFile)
    predOutputName = "{}_pred.txt".format(outFile)
    w = []
    with open(trueOutputName, 'w') as outputTrue,\
            open(predOutputName, 'w') as outputPred:
        inputData = Data.fromFiles([file])
        for d in inputData:
            assert isinstance(d, Data.Element)
            obs = d.observed
            hid = d.hidden
            sa = ViterbiHatSequenceAnalyzer(hmm, obs)
            pred = sa.getTrace()
            outputTrue.write(">{}\n{}\n#{}\n".format(d.name, obs, hid))
            outputPred.write(">{}\n{}\n#{}\n".format(d.name, obs, pred))
            w.append(len(hid))
    return (trueOutputName, predOutputName)

def learnAndPrintModel(fileNames, model):
    d = Data.fromFiles(fileNames)
    
    N = len(d.observableStates)
    if( model == '3State'):
        hidden = ['i','M','o']
        cA = getAConstraints(hidden,["io","oi"])
        cP = getPiConstraints(hidden,["M"])
    elif( model == '4State'):
        hidden = ['i','I','O','o'] # read as (i)nside, (I)nwards, (O)utwards, (o)utside
        cA = getAConstraints(hidden,["io","oi","IO","OI","iI","oO","Oi","Io"])
        cP = getPiConstraints(hidden,["I","O"])
                    
    fun = random            #random starting guess
    #fun = lambda : 1.0      #uniform starting guess
    piStart, AStart, phiStart = getUnnormalizedStartingGuess(fun,len(hidden),N,piConstraints=cP, aConstraints=cA)
    piStart[0] = 0.49
    piStart[len(piStart)-1] = 0.51
        
    # This should also work with LogSumSequenceAnalyzer
    sa = ScaledPosteriorSequenceAnalyzer
    #sa = LogSumSequenceAnalyzer
    hT = PosteriorTrainer(sa, Hmm(hidden,d.observableStates,piStart,AStart,phiStart))
        
    hT.dump('initial-parameters-'+model+'.txt')
    hT.train(d.getObserved(),tol=1e-4,maxIt=1000)
    hT.dump('final-parameters-'+model+'.txt')
    hT.hmm.printRepr()
    return hT.hmm

if __name__ == '__main__':
    for model in ['3State','4State']:
        cv = "task8_CV_"+model
        if not os.path.exists(cv):
            os.mkdir(cv)
        acs = []
        testNames = []
        for i in range(10):
            fileNames = ['set160.{}.labels.txt'.format(x) for x in range(10)]
            testFile = fileNames[i]
            testNames.append(testFile)
            del fileNames[i]
            fn = [os.path.join("Dataset160",f) for f in fileNames]
            hmm = learnAndPrintModel(fn, model)
            outfile = os.path.join(cv,"{}_task8_{}".format(testFile[0:-4], i))
            (t,p) = predict(hmm, os.path.join("Dataset160", testFile), outfile)
            proc = Popen(["python2", "compare_tm_pred.py", t, p], stdout=PIPE)
            out = str(proc.communicate()[0])[2:-1]
            aOut = [ o for o in out.split('\\n') if not o == '' ]
            print(aOut)
            z = zip(*[iter(aOut)] * 2)
            localAc = []
            for name, result in z:
                assert isinstance(name, str)
                m = re.search("AC = (.+)", result)
                ac = float('nan')
                if m:
                    ac = float(m.group(1))
                # print("{} : {}".format(name, ac))
                if math.isfinite(ac) and not math.isnan(ac):
                    if name.startswith("Summary"):
                        acs.append(ac)
                    else:
                        localAc.append(ac)
            print(testFile)

        print("Summary of individual folds for "+model+":")
        print("\n".join( [ "fold{} AC : {}".format(i, acs[i]) for i in range(len(acs))]))
        print("Mean and Var over all folds")
        print("Mean AC  : {}".format(nanmean(acs)))
        print("Var AC   : {}".format(nanvar(acs)))
        print("Std AC   : {}".format(nanstd(acs)))
