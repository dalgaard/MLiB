import time
from hmmTools import *
import hmmToolsMaster
def getAnalysis(f,sequenceAnalyzer):
    trace=sequenceAnalyzer.getTrace()
    return "# "+trace+"\n; log P(x,z) = "+str(sequenceAnalyzer.logLikelihood(trace))+"\n\n"
            
h = Hmm('hmm-tm.txt')

post = open("posterior.txt",'w')

post.write("Posterior-decoding using the scaled forward and backward algorithms\n")


with open('test-sequences-project2.txt','r') as f:
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.startswith(">"):
            for f in [post]:
                f.write(line.strip()+"\n")
                f.write(lines[iline+1].strip()+"\n")
            # the scaled posterior and the viterbi decodings are calculated at the same time
            sN = ScaledPosteriorSequenceAnalyzer(h,lines[iline+1].strip(),1)
            sO = hmmToolsMaster.ScaledPosteriorSequenceAnalyzer(h,lines[iline+1].strip())
            post.write(getAnalysis(post,ScaledPosteriorSequenceAnalyzer(h,lines[iline+1].strip(),1)))

post.close()

