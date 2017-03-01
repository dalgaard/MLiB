#!/usr/bin/python

import sys
from hmmTools import Hmm, ViterbiHatSequenceAnalyzer

def main(argv):
    if len(argv) != 2:
        print("arguments : <hmm specification file> <observed sequence>\n")
        sys.exit(1)
    sa = ViterbiHatSequenceAnalyzer(Hmm.fromFile(argv[0]), argv[1])
    trace = sa.getTrace()
    ll = sa.logLikelihood(trace)
    print("Trace         : {}".format(trace))
    print("LogLikelihood : {}".format(ll))

if __name__ == "__main__":
   main(sys.argv[1:])
