#!/usr/bin/python

import sys
from hmmTools import Hmm, ViterbiHatSequenceAnalyzer, ScaledPosteriorSequenceAnalyzer

def processviterbi(content, hmm):
    viterbi_output_txt = 'viterbi-output.txt'
    posterior_output_txt = 'posterior-output.txt'
    with open(viterbi_output_txt, 'w') as fViterbi:
        with open(posterior_output_txt, 'w') as fPosterior:
            fPosterior.write("; Posterior-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n");
            fViterbi.write("; Viterbi-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n")
            i = 0
            while i < len(content):
                l = content[i]
                if len(l) > 0 and l[0] == '>':
                    fViterbi.write(l+'\n')
                    fPosterior.write(l+'\n')
                    i += 1
                    obs = content[i]
                    fViterbi.write(obs+'\n#\n')
                    fPosterior.write(obs+'\n#\n')
                    i += 1
                    saVit = ViterbiHatSequenceAnalyzer(hmm, obs)
                    traceVit = saVit.getTrace()
                    llVit = saVit.logLikelihood(traceVit)
                    posteriorVit = saVit.getPosterior()
                    fViterbi.write(traceVit+'\n')
                    fViterbi.write('; log P(x,z) (as computed by Viterbi) = {}\n'.format(posteriorVit))
                    fViterbi.write('; log P(x,z) (as computer by your log-joint-prob) = {}\n\n'.format(llVit))
                    saPost = ScaledPosteriorSequenceAnalyzer(hmm, obs)
                    tracePost = saPost.getTrace()
                    llPost = saPost.logLikelihood(tracePost)
                    fPosterior.write(tracePost+'\n')
                    fPosterior.write('; log P(x,z) (as computer by your log-joint-prob) = {}\n\n'.format(llPost))
                else:
                    i += 1
    print('output written to {} and {}'.format(viterbi_output_txt, posterior_output_txt))



def main(argv):
    if len(argv) != 1:
        print("arguments : <hmm specification file>\n")
        sys.exit(1)
    with open("sequences-project2.txt",'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content if x.strip() != '']
    hmm = Hmm(argv[0])
    processviterbi(content, hmm)

if __name__ == "__main__":
   main(sys.argv[1:])
