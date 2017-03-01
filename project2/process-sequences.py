#!/usr/bin/python

import sys
from hmmTools import Hmm, ViterbiHatSequenceAnalyzer, ScaledPosteriorSequenceAnalyzer, ViterbiSequenceAnalyzer, LogSumSequenceAnalyzer

def processviterbi(content, hmm):
    viterbi_log_output_txt = 'viterbi-output.txt'
    viterbi_scaled_output_txt = 'viterbi-scaled-output.txt'
    posterior_log_output_txt = 'posterior-output.txt'
    posterior_scaled_output_txt = 'posterior-scaled-output.txt'
    with open(viterbi_log_output_txt, 'w') as fVitLog, \
            open(posterior_log_output_txt, 'w') as fPostLog, \
            open(posterior_scaled_output_txt, 'w') as fPostScaled, \
            open(viterbi_scaled_output_txt, 'w') as fVitScaled:
        fPostLog.write("; Posterior-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n")
        fPostScaled.write("; Posterior-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n")
        fVitLog.write("; Viterbi-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n")
        fVitScaled.write("; Viterbi-decodings of sequences-project2.txt using HMM hmm-tm.txt\n\n")
        i = 0
        while i < len(content):
            l = content[i]
            if len(l) > 0 and l[0] == '>':
                fVitLog.write(l+'\n')
                fVitScaled.write(l+'\n')
                fPostLog.write(l+'\n')
                fPostScaled.write(l+'\n')
                i += 1
                obs = content[i]
                fVitLog.write(obs+'\n#\n')
                fVitScaled.write(obs+'\n#\n')
                fPostLog.write(obs+'\n#\n')
                fPostScaled.write(obs+'\n#\n')
                i += 1
                doViterbi(fVitLog, hmm, obs, True)
                doViterbi(fVitScaled, hmm, obs, False)
                doPosterior(fPostLog, hmm, obs, True)
                doPosterior(fPostScaled, hmm, obs, False)
            else:
                i += 1
    print('output written to files:{}, {}, {}, {}'.format(viterbi_log_output_txt, viterbi_scaled_output_txt, posterior_log_output_txt, posterior_scaled_output_txt))


def doPosterior(f, hmm, obs, ln=False):
    if ln:
        saPost = LogSumSequenceAnalyzer(hmm, obs)
    else :
        saPost = ScaledPosteriorSequenceAnalyzer(hmm, obs)
    tracePost = saPost.getTrace()
    llPost = saPost.logLikelihood(tracePost)
    f.write(tracePost + '\n')
    f.write('; log P(x,z) (as computer by your log-joint-prob) = {}\n\n'.format(llPost))


def doViterbi(fVitLog, hmm, obs, ln=False):
    if ln:
        saVit = ViterbiHatSequenceAnalyzer(hmm, obs)
    else:
        saVit = ViterbiSequenceAnalyzer(hmm, obs)
    traceVit = saVit.getTrace()
    llVit = saVit.logLikelihood(traceVit)
    posteriorVit = saVit.getPosterior()
    fVitLog.write(traceVit + '\n')
    fVitLog.write('; log P(x,z) (as computed by Viterbi) = {}\n'.format(posteriorVit))
    fVitLog.write('; log P(x,z) (as computer by your log-joint-prob) = {}\n\n'.format(llVit))


def main(argv):
    if len(argv) != 1:
        print("arguments : <hmm specification file>\n")
        sys.exit(1)
    with open("sequences-project2.txt",'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content if x.strip() != '']
    hmm = Hmm.fromFile(argv[0])
    processviterbi(content, hmm)

if __name__ == "__main__":
   main(sys.argv[1:])
