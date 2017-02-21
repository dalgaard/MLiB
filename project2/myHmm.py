from hmmTools import *
def getAnalysis(f,sequenceAnalyzer,traceType):
    trace=sequenceAnalyzer.getTrace(traceType)
    return "# "+trace+"\n; log P(x,z) = "+str(sequenceAnalyzer.logLikelihood(trace))+"\n\n"
            
h = Hmm('hmm-tm.txt')

post = open("posterior.txt",'w')
logPost = open("logPost.txt",'w')
viterbi = open("viterbi.txt",'w')

post.write("Posterior-decoding using the scaled forward and backward algorithms\n")
logPost.write("Posterior-decoding using the logsum in the forward and backward algorithms\n")
viterbi.write("Viterbi-decoding\n")

with open('test-sequences-project2.txt','r') as f:
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.startswith(">"):
            for f in [post,logPost,viterbi]:
                f.write(line.strip()+"\n")
                f.write(lines[iline+1].strip()+"\n")
            # the scaled posterior and the viterbi decodings are calculated at the same time
            sa = HmmSequenceAnalyzer(h,lines[iline+1].strip())
            post.write(getAnalysis(post,sa,"posterior"))
            viterbi.write(getAnalysis(post,sa,"viterbi"))
            # for the log posterior we need to initialize differently
            logPost.write(getAnalysis(logPost,HmmSequenceAnalyzer(h,lines[iline+1].strip(),logVersion=True),"logPosterior"))

post.close()
logPost.close()
viterbi.close()

