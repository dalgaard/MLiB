import time
from hmmTools import *
def getAnalysis(f,sequenceAnalyzer):
    trace=sequenceAnalyzer.getTrace()
    return "# "+trace+"\n; log P(x,z) = "+str(sequenceAnalyzer.logLikelihood(trace))+"\n\n"
            
h = Hmm.fromFile('hmm-tm.txt')

post = open("posterior.txt",'w')
logPost = open("logPost.txt",'w')
viterbi = open("viterbi.txt",'w')

post.write("Posterior-decoding using the scaled forward and backward algorithms\n")
logPost.write("Posterior-decoding using the logsum in the forward and backward algorithms\n")
viterbi.write("Viterbi-decoding\n")

vT = 0.0
sP = 0.0
lS = 0.0

with open('test-sequences-project2.txt','r') as f:
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.startswith(">"):
            for f in [post,logPost,viterbi]:
                f.write(line.strip()+"\n")
                f.write(lines[iline+1].strip()+"\n")
            # the scaled posterior and the viterbi decodings are calculated at the same time
            t1 = time.process_time()
            post.write(getAnalysis(post,ScaledPosteriorSequenceAnalyzer(h,lines[iline+1].strip())))
            t2 = time.process_time()
            viterbi.write(getAnalysis(post,ViterbiSequenceAnalyzer(h,lines[iline+1].strip(),1)))
            t3 = time.process_time()
            logPost.write(getAnalysis(logPost,LogSumSequenceAnalyzer(h,lines[iline+1].strip(),1)))
            t4 = time.process_time()
            sP += t2-t1
            vT += t3-t2
            lS += t4-t3
print("Scaled :",sP)
print("Viterbi :",vT)
print("LogSum :",lS)

post.close()
logPost.close()
viterbi.close()

