#!/usr/bin/python

import sys
from hmmTools import Hmm, HmmSequenceAnalyzer

def main(argv):
    if len(argv) != 3:
        print("arguments : <hmm specification file> <observables> <hidden>\n")
        sys.exit(1)
    sa = HmmSequenceAnalyzer(Hmm.fromFile(argv[0]), argv[1])
    ll = sa.logLikelihood(argv[2])
    print("LogLikelihood : {}".format(ll))

if __name__ == "__main__":
   main(sys.argv[1:])
