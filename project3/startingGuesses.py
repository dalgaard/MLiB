
def getUnnormalizedStartingGuess3State(f,K,N):
    piStart  =  [ f() for i in range(K) ]
    
    AStart   = [[ f() for i in range(K) ] for j in range(K)]
    # set the io and oi transition probabilities to 0
    AStart[0][K-1] = 0.0 
    AStart[K-1][0] = 0.0
    phiStart = [[ f() for i in range(N) ] for j in range(K)]
    return piStart,AStart,phiStart

def getUnnormalizedStartingGuess4State(f,K,N):
    # get a well shaped initial starting guess
    piStart  =  [ f() for i in range(K) ]
    
    AStart   = [[ f() for i in range(K) ] for j in range(K)]
    # set the io/IO/iO and oi/OI/oI transition probabilities to 0
    AStart[0][K-1] = 0.0 # no io
    AStart[K-1][0] = 0.0 # no oi
    AStart[1][2] = 0.0 # no IO
    AStart[2][1] = 0.0 # no OI
    AStart[0][1] = 0.0 # no iI
    AStart[3][2] = 0.0 # no oO
    AStart[2][0] = 0.0 # no Oi
    AStart[1][3] = 0.0 # no Io
    
    phiStart = [[ f() for i in range(N) ] for j in range(K)]
    
    return piStart,AStart,phiStart
