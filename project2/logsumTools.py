import math

def exp(a):
    return math.exp(a)
def log(a):
    return math.log(a) if a > 1e-13 else float("-inf")
def logsum(logX,logY):
    return logX if logY==float("-inf") else logY if logX==float("-inf") else logX+log(1+math.exp(logY-logX)) if logX>logY else logY+log(1+math.exp(logX-logY)) 
#def logsum(*logArgs):
#    s = float("-inf")
#    for arg in logArgs:
#        s = logsum(s,arg)
#    return s


