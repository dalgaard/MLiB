import unittest
from hmmTools import Hmm, HmmSequenceAnalyzer
from math import log

expectedHidden = [0.496359, 0.188095, 0.315546]
expectedTransitions = \
    [[0.990971, 0.009029, 0.000000],
     [0.023083, 0.953090, 0.023827],
     [0.000000, 0.013759, 0.986241]]
expectedEmissions = \
    [[0.043601,0.011814,0.053446,0.065541,0.049508,0.049789,0.054571,0.024191,0.055977,0.035162,0.103235,0.045007,0.029536,0.048101,0.075105,0.059634,0.068354,0.016315,0.067792,0.043319],
     [0.102010,0.019360,0.009680,0.011914,0.033507,0.103500,0.118392,0.003723,0.000745,0.039464,0.138496,0.014147,0.011914,0.026806,0.067014,0.012658,0.073716,0.037230,0.119136,0.056590],
     [0.082374,0.008415,0.059345,0.059345,0.069973,0.031001,0.049159,0.019043,0.081045,0.025244,0.068202,0.047830,0.032772,0.052259,0.073959,0.086802,0.056244,0.007086,0.062445,0.027458]]


class HmmLoaderTestCase(unittest.TestCase):

    def setUp(self):
        self.hmm = Hmm('hmm-tm.txt')

    def test_hidden(self):
        self.assertEqual(self.hmm.hidden, ['i','M', 'o'], 'incorrect hidden states')

    def test_observables(self):
        self.assertEqual(self.hmm.observables, ['A', 'C', 'E' ,'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'])

    def test_pi(self):
        self.assertEqual(self.hmm.pi, expectedHidden)

    def test_transitions(self):
        self.assertEqual(self.hmm.A ,expectedTransitions, 'incorrect transitions')

    def test_emissions(self):
        self.assertEqual(self.hmm.emissions, expectedEmissions, 'incorrect emissions')

    def test_ll(self):
        hid = ['M', 'o', 'o']
        obs = ['A', 'C', 'E']
        expected = log(0.188095) + log(0.102010) + log(0.023827) + log(0.008415) + log(0.986241) + log(0.059345)
        sa = HmmSequenceAnalyzer(self.hmm,obs)
        self.assertEqual(sa.logLikelihood(hid), expected)

    def test_ll_large(self):
        sa = HmmSequenceAnalyzer(self.hmm,list("MAKNLILWLVIAVVLMSVFQSFGPSESNGRKVDYSTFLQEVNNDQVREARINGREINVTKKDSNRYTTYIPVQDPKLLDNLLTKNVKVVGEPPEEPSLLASIFISWFPMLLLIGVWIFFMRQMQGGGGKGAMSFGKSKARMLTEDQIKTTFADVAGCDEAKEEVAELVEYLREPSRFQKLGGKIPKGVLMVGPPGTGKTLLAKAIAGEAKVPFFTISGSDFVEMFVGVGASRVRDMFEQAKKAAPCIIFIDEIDAVGRQRGAGLGGGHDEREQTLNQMLVEMDGFEGNEGIIVIAATNRPDVLDPALLRPGRFDRQVVVGLPDVRGREQILKVHMRRVPLAPDIDAAIIARGTPGFSGADLANLVNEAALFAARGNKRVVSMVEFEKAKDKIMMGAERRSMVMTEAQKESTAYHEAGHAIIGRLVPEHDPVHKVTIIPRGRALGVTFFLPEGDAISASRQKLESQISTLYGGRLAEEIIYGPEHVSTGASNDIKVATNLARNMVTQWGFSEKLGPLLYAEEEGEVFLGRSVAKAKHMSDETARIIDQEVKALIERNYNRARQLLTDNMDILHAMKDALMKYETIDAPQIDDLMARRDVRPPAGWEEPGASNNSGDNGSPKAPRPVDEPRTPNPGNTMSEQLGDK"))
        self.assertAlmostEqual(sa.logLikelihood(list("ooooMMMMMMMMMMMMMMMooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooMMMMMMMMMMMMMMMMMMMMMMMoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")),
            -1849.073368,
            6)

def test_generator(obs, expectedTrace, expectedPosterior, method="posterior"):
    def test(self):
        sa = HmmSequenceAnalyzer(self.hmm, obs)
        trace = sa.getTrace(method)
        posterior = sa.logLikelihood(trace)
        self.assertEqual(trace, expectedTrace)
        self.assertAlmostEqual(posterior, expectedPosterior, 6)
    return test


def generateTest(content, method):
    for i in range(len(content)):
        l = content[i]
        if len(l) > 0 and l[0] == '>':
            test_name = 'test_{}_{}'.format(method, l[1:])
            i += 1
            obs = content[i]
            i += 1
            hid = content[i][2:]
            i += 1
            posterior = float(content[i][15:])
            i += 1
            test = test_generator(obs, hid, posterior, method)
            setattr(HmmLoaderTestCase, test_name, test)
        else:
            i += 1


if __name__ == '__main__':
    with open("test-sequences-project2-posterior-output.txt",'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    generateTest(content, 'posterior')
    with open("test-sequences-project2-viterbi-output.txt",'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    generateTest(content, 'viterbi')
    unittest.main()
