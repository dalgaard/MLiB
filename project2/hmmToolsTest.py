import unittest
from hmmTools import Hmm
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
        self.assertEqual(self.hmm.logLikelihood(hid, obs), expected)

    def test_ll_large(self):
        self.assertAlmostEqual(self.hmm.logLikelihood(
            list("ooooMMMMMMMMMMMMMMMooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooMMMMMMMMMMMMMMMMMMMMMMMoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"),
            list("MAKNLILWLVIAVVLMSVFQSFGPSESNGRKVDYSTFLQEVNNDQVREARINGREINVTKKDSNRYTTYIPVQDPKLLDNLLTKNVKVVGEPPEEPSLLASIFISWFPMLLLIGVWIFFMRQMQGGGGKGAMSFGKSKARMLTEDQIKTTFADVAGCDEAKEEVAELVEYLREPSRFQKLGGKIPKGVLMVGPPGTGKTLLAKAIAGEAKVPFFTISGSDFVEMFVGVGASRVRDMFEQAKKAAPCIIFIDEIDAVGRQRGAGLGGGHDEREQTLNQMLVEMDGFEGNEGIIVIAATNRPDVLDPALLRPGRFDRQVVVGLPDVRGREQILKVHMRRVPLAPDIDAAIIARGTPGFSGADLANLVNEAALFAARGNKRVVSMVEFEKAKDKIMMGAERRSMVMTEAQKESTAYHEAGHAIIGRLVPEHDPVHKVTIIPRGRALGVTFFLPEGDAISASRQKLESQISTLYGGRLAEEIIYGPEHVSTGASNDIKVATNLARNMVTQWGFSEKLGPLLYAEEEGEVFLGRSVAKAKHMSDETARIIDQEVKALIERNYNRARQLLTDNMDILHAMKDALMKYETIDAPQIDDLMARRDVRPPAGWEEPGASNNSGDNGSPKAPRPVDEPRTPNPGNTMSEQLGDK")),
            -1849.073368,
            6)
