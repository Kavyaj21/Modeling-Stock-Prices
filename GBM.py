import numpy as np
import matplotlib.pyplot as pl

def GeneratePaths(spot, process, maturity, nSteps, nPaths, correlation = None):
    dt = maturity / nSteps
    
    if (isinstance(correlation, np.ndarray)):
        nProcesses = process.shape[0]
        result = np.zeros(shape = (nProcesses, nPaths, nSteps))
        
        # loop through number of paths
        for i in range(nPaths):
            choleskyMatrix = np.linalg.cholesky(correlation)
            e = np.random.normal(size = (nProcesses, nSteps))            
            paths = np.dot(choleskyMatrix, e)
            for j in range(nSteps):
                for k in range(nProcesses):
                    if(j == 0):
                        result[k, i, j] = paths[k, j] = spot[k]
                    else:
                        result[k, i, j] = paths[k, j] = process[k](paths[k, j - 1], dt, paths[k, j])

    else:
        result = np.zeros(shape = (1, nPaths, nSteps))
        for i in range(nPaths):
            path = np.random.normal(size = nSteps)
            result[0, i, 0] = path[0] = spot
            for j in range(nSteps):
                if(j > 0):
                    result[0, i, j] = path[j] = process(path[j - 1], dt, path[j])
    return result

mu = 0.03
sigma = 0.25

BrownianMotion = lambda s, dt, e: s + mu * s * dt + sigma * s * np.sqrt(dt) * e   
maturity = 1.0
nPaths = 10
nSteps = 250

matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
spots = np.array([100.0, 100.0])
processes = np.array([BrownianMotion, BrownianMotion])
MultiAssetPaths = GeneratePaths(spots, processes, maturity, nSteps, nPaths, matrix)
f, subPlots = pl.subplots(processes.shape[0], sharex = True)
for i in range(processes.shape[0]): 
    for j in range(nPaths):
        subPlots[i].plot(MultiAssetPaths[i, j, :])
pl.show()