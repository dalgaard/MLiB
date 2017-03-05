import os
from collections import namedtuple
from random import shuffle

class Data(object):

    __slots__ = ['data',
                 'hiddenStates',
                 'observableStates']

    Element = namedtuple("Element", "name observed hidden")

    def __init__(self, data, hiddenStates, observableStates):
        self.data = data
        self.hiddenStates = hiddenStates
        self.observableStates = observableStates

    @classmethod
    def fromFiles(cls, filenames):
        data = []
        labelSet = set()
        obsSet = set()
        for fn in filenames:
            # print('reading {}'.format(fn))
            with open(fn, 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            i=0
            while i < len(content):
                l = content[i]
                if len(l) > 0 and l[0] == '>':
                    n = l[1:]
                    i += 1
                    obs = content[i]
                    i += 1
                    hid = content[i][1:].strip()
                    labelSet.update(hid)
                    obsSet.update(obs)
                    data.append(Data.Element(name = n, observed = obs, hidden = hid))
                i += 1
        hiddenStates = [ x for x in labelSet ]
        observableStates = [ x for x in obsSet ]
        return(cls(data, hiddenStates, observableStates))
    
    def getObserved(self):
        return [ d.observed for d in self.data ]

    def __iter__(self):
        for d in self.data:
            yield d

class Counts(object):

    __slots__ = ['piCount',
                 'transitionCount',
                 'emissionCount']

    def __init__(self, data):
        self.piCount = dict()
        self.transitionCount = dict()
        self.emissionCount = dict()
        for e in data.data:
            obs = e.observed
            hid = e.hidden
            for src, dest in zip(hid, hid[1:]):
                trans = (src, dest)
                self.transitionCount[trans] = self.transitionCount.get(trans, 0) + 1
            for h, o in zip(hid, obs):
                e = (h, o)
                self.emissionCount[e] = self.emissionCount.get(e, 0) + 1
            self.piCount[hid[0]] = self.piCount.get(hid[0], 0) + 1

def kFoldGenerator(k, allData):
    data = allData.data
    shuffle(data)
    if k > len(data) or k < 2 :
        return
    testSize = len(data) // k
    print("testSize={}".format(testSize))
    for i in range(k):
        chunkStart = i * testSize
        chunkEnd = chunkStart + testSize
        test = data[chunkStart:chunkEnd]
        train = data[0:chunkStart]
        train.extend(data[chunkEnd:len(data)])
        yield (train, test)

if __name__ == '__main__':
    fileNames = [os.path.join('Dataset160','set160.{}.labels.txt'.format(x)) for x in range(10)]
    d = Data.fromFiles(fileNames)
    c = Counts(d)
    for h in d.hiddenStates:
        print('{} : {}'.format(h, c.piCount.get(h, 'NA')))
    for src in d.hiddenStates:
        for dest in d.hiddenStates:
            print('trans {}->{} : {}'.format(src, dest, c.transitionCount.get((src, dest), 0)))
    for h in d.hiddenStates:
        for o in d.observableStates:
            print('emis {}->{} {}'.format(h, o, c.emissionCount.get((h,o), 0)))
    for train, test in kFoldGenerator(10, d):
        print('test:  {}\ntrain: {}'.format(",".join([t.name for t in test]), ",".join([t.name for t in train]) ))
