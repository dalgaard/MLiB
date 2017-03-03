import sys
sys.path.append('../project2')
from hmmTools import *

def getAnalysis(f,sequenceAnalyzer):
    trace=sequenceAnalyzer.getTrace()
    return "# "+trace+"\n; log P(x,z) = "+str(sequenceAnalyzer.logLikelihood(trace))+"\n\n"

def testAgainstProj2(hmm, fname = "compare-project2.txt"):
    with open('../project2/test-sequences-project2.txt','r') as f:
        post = open(fname,'w')
        post.write("Posterior-decoding and viterbi decodings using the scaled forward and backward algorithms\n")
        lines = f.readlines()
        for iline,line in enumerate(lines):
            if line.startswith(">"):
                for f in [post]:
                    f.write(line.strip()+"\n")
                    f.write(lines[iline+1].strip()+"\n")
                # the scaled posterior and the viterbi decodings are calculated at the same time
                post.write(getAnalysis(post,ScaledPosteriorSequenceAnalyzer(hmm,lines[iline+1].strip())))
                post.write(getAnalysis(post,ViterbiSequenceAnalyzer(hmm,lines[iline+1].strip())))
