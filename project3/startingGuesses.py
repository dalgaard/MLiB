
def getPiConstraints(hidden,constr):
    cA = []
    for i,src in enumerate(hidden):
        if src in constr:
            cA.append(i)
    return cA
def getAConstraints(hidden,constr):
    cA = []
    for i,src in enumerate(hidden):
        for j,dest in enumerate(hidden):
            if src+dest in constr:
                cA.append([i,j])
    return cA
def getPhiConstraints(hidden,observable,constr):
    cA = []
    for i,src in enumerate(hidden):
        for j,dest in enumerate(observable):
            if src+dest in constr:
                cA.append([i,j])
    return cA

def getUnnormalizedStartingGuess(f,K,N,piConstraints=[],aConstraints=[],phiConstraints=[]):
    # get a well shaped initial starting guess
    piStart  =  [ f() for i in range(K) ]
    for c in piConstraints:
        piStart[c] = 0.0
    
    AStart   = [[ f() for i in range(K) ] for j in range(K)]
    for i,j in aConstraints:
        AStart[i][j] = 0.0
    
    phiStart = [[ f() for i in range(N) ] for j in range(K)]
    for i,j in phiConstraints:
        phiStart[i][j] = 0.0
    
    return piStart,AStart,phiStart
