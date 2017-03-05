from Data import Data
from hmmTools import ViterbiHatSequenceAnalyzer
from subprocess import Popen, PIPE
import math
import os
import re
from numpy import nanmean, nanstd, nanvar
from task2 import learnAndPrint4StateModel

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
            tmpPred = sa.getTrace()
            pred = "".join([s if s == 'i' or s == 'o' else 'M' for s in tmpPred ])
            outputTrue.write(">{}\n{}\n#{}\n".format(d.name, obs, hid))
            outputPred.write(">{}\n{}\n#{}\n".format(d.name, obs, pred))
            w.append(len(hid))
    return (trueOutputName, predOutputName)

if __name__ == '__main__':
    cv = "task4_CV"
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
        hmm = learnAndPrint4StateModel(fn, printModel=False)
        outfile = os.path.join(cv,"{}_task3_{}".format(testFile[0:-4], i))
        (t,p) = predict(hmm, os.path.join("Dataset160", testFile), outfile)
        proc = Popen(["python", "compare_tm_pred.py", t, p], stdout=PIPE)
        out = str(proc.communicate()[0])[2:-1]
        aOut = [ o for o in out.split('\\n') if not o == '' ]
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

    print("Summary of individual folds:")
    print("\n".join( [ "fold{} AC : {}".format(i, acs[i]) for i in range(len(acs))]))
    print("Mean and Var over all folds")
    print("Mean AC  : {}".format(nanmean(acs)))
    print("Var AC   : {}".format(nanvar(acs)))
    print("Std AC   : {}".format(nanstd(acs)))