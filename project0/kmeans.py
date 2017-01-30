import numpy as np
import numpy.linalg as npl
from random import randrange,random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getStartingGuess(X,K):
    #return [ X[randrange(len(X))] for i in range(K) ]
    centers = np.array([X[randrange(len(X))]])
    d=computeDistanceDistribution(X,centers)
    for i in range(K-1):
        c=X[np.random.choice(range(len(X)), 1, d)]
        centers=np.vstack((centers, c))
        d=computeDistanceDistribution(X, centers)
    print("### starting guess: ",centers)
    return centers

def computeDistanceDistribution(data, clusterCenters):
    result=[]
    for x in data:
        d=[ npl.norm(x-c)**2 for c in clusterCenters ]
        result.append(min(d))
    return [ r / sum(result) for r in result ]

def findClusters(X,K,plot='n'):
    if plot not in [ 'n', 'i','f']:
        print("invalid option for argument plot: ",plot)
        exit()
        
    M = getStartingGuess(X,K)
    converged = False
    N = len(X)
    P = len(X[0])
    Jprev = 0.0
    niter = 0
    print( "### it\tJ\tΔJ")
    while(not converged):
        # get the cluster associations
        A = [ [ 0 for k in range(K) ] for n in range(N) ]
        J = 0.0
        for ix,x in enumerate(X):
            
            #init
            minD = npl.norm(x - M[0])**2
            minK = 0
            
            # look in all cluster prototypes for a best match
            for im,m in enumerate(M):
                D = npl.norm(x-m)**2
                if( D < minD ):
                    minD = D
                    minK = im
            
            A[ix][minK] = 1
            J += minD
        
        if(plot == 'i'):
            plotData(X,M,A)
        
        # get the new means 
        for k in range(K):
            n = 0
            mu = np.array([ 0.0 for n in range(P) ])
            for ix,x in enumerate(X):
                mu += A[ix][k] * x
                n += A[ix][k]
            M[k] = mu/n
        
        Jdiff = abs(J-Jprev)
        print( "### "+str(niter)+"\t"+"{0:.2g}".format(J)+"\t"+"{0:.2g}".format(Jdiff))
        converged = Jdiff < 1.0e-13 or niter > 200
        Jprev = J
        niter += 1
        
    print ("Final associations:")
    for k in range(K):
        print(" "+str(k)+": ",end="")
        for ix in range(N):
            if( A[ix][k] == 1):
                print(str(ix+1)+" ",end="")
        print("")
    if(plot in ['i','f']):
        plotData(X,M,A)
        plt.show()
    return A,M
        
def plotData(X,M,A):
    fig = plt.figure()
    N = len(X)
    P = len(X[0])
    K = len(M)
    if P>2:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    C = np.dot(X.transpose(),X)/(N-1)
    e,V = npl.eig(C)
    V = np.real(V)
    thr = 1.0e-6
    print(" Eigenvalues > "+"{0:.1g}".format(thr)+": ",end="")
    i = 0
    while(e[i] > thr):
        print("{0:.2g}".format(e[i]),end="\n" if i+1>=P else ", " if (e[i+1] > thr) else "\n")
        i+=1
        if(i>=P):
            break
    
    if( False ):
        #plot the first principal components
        XPCoA = np.dot(X,V)
        MPCoA = np.dot(M,V)
        xc = 0
        yc = 1
        zc = 2 if P>2 else -1
    else:
        #plot the dimensions which are "most colinear" with the principal components
        XPCoA = X
        MPCoA = M
        xc = np.argmax(V[0])
        yc = np.argmax(V[1])
        zc = np.argmax(V[3]) if P>2 else -1
        print("Dims: "+str(xc)+" "+str(yc)+" "+str(zc))
            
    

    
    cols = ['r','b','k','g','c','m','y']
    for k in range(K):
        c = cols[k] if k<len(cols) else random()
        x = [float(MPCoA[k][xc])]
        y = [float(MPCoA[k][yc])]
        z = [float(MPCoA[k][zc]) if P>2 else 0]
        if( P>2 ):
            ax.scatter(x,y,z,marker='x',c=c)
        else:
            ax.scatter(x,y,marker='x',c=c)
        x = []
        y = []
        z = []
        for n in range(N):
            if(A[n][k] == 1):
                x.append(float(XPCoA[n][xc]))
                y.append(float(XPCoA[n][yc]))
                z.append(float(XPCoA[n][zc]) if P>2 else 0 )
        if( P>2 ):
            ax.scatter(x,y,z,marker='o',c=c)
        else:
            ax.scatter(x,y,marker='o',c=c)
        
    
def centerColumns(X):
    N = len(X)
    P = len(X[0])
    mu = [ 0.0 for i in range(P) ]
    
    for p in range(P):
        for n in range(N):
            mu[p] += X[n][p] / N
    
    for n in range(N):
        for p in range(P):
            X[n][p] = (X[n][p] - mu[p])/mu[p]
    return X

def normalizeColumns(X):
    N = len(X)
    P = len(X[0])
    mu = [ 0.0 for i in range(P) ]
    si = [ 0.0 for i in range(P) ]
    
    for p in range(P):
        for n in range(N):
            mu[p] += X[n][p] / N
        for n in range(N):
            si[p] += (X[n][p] - mu[p])**2/(N-1)
        si[p] = (si[p])**(0.5)
    
    for n in range(N):
        for p in range(P):
            X[n][p] = (X[n][p] - mu[p])/si[p] 
    return X
