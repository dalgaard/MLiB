import os
from collections import namedtuple


class Data(object):

    __slots__ = ['data', 'piCount', 'transitionCount', 'hiddenStates', 'observableStates']

    Element = namedtuple("DataElement", "name observed hidden")

    def __init__(self):
        data = []
        piCount = dict()
        transitionCount = dict()
        for root, dirs, filenames in os.walk('Dataset160'):
            for fn in filenames:
                print('reading {}'.format(fn))
                with open(os.path.join('Dataset160',fn), 'r') as f:
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
                        for src, dest in zip(hid, hid[1:]):
                            trans = (src, dest)
                            transitionCount[trans] = transitionCount.get(trans, 0) + 1
                        piCount[hid[0]] = piCount.get(hid[0], 0) + 1
                        data.append(self.Element(name = n, observed = obs, hidden = hid))
                    i += 1
        self.data = data
        self.piCount = piCount
        self.transitionCount = transitionCount
        s = set()
        for src, dest in transitionCount:
            s.add(src)
            s.add(dest)
        self.hiddenStates = [ x for x in s ]

if __name__ == '__main__':
    d = Data()
    for x in d.piCount:
        print(x,':',d.piCount[x])
    for h in d.hiddenStates:
        print(h)
    for src in d.hiddenStates:
        for dest in d.hiddenStates:
            print('{}->{} {}'.format(src, dest, d.transitionCount.get((src, dest), 'missing')))