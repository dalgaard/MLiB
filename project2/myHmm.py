from hmmTools import *
h = Hmm('hmm-tm.txt')

with open('test-sequences-project2.txt','r') as f:
    lines = f.readlines()
    for iline,line in enumerate(lines):
        if line.startswith(">"):
            sa = HmmSequenceAnalyzer(h,lines[iline+1].strip())
            print(line.strip())
            print(lines[iline+1].strip())
            trace=sa.getTrace("viterbi")
            print("# "+trace)
            print("; log P(x,z) =",sa.logLikelihood(trace))
            print()

