import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def extract(x):
    res=np.random.rand(10,2)
    return res

#function to optimze
def f(x,y):
    x=extract(x)
    return (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options)

# want x to be close to 3,4
y=np.tile([3,4], [10,1])
print(y)

# each particle is 3d, each particle is then embedded to be 2 d and optimized to be close to 3,4

# Perform optimization
cost, pos = optimizer.optimize(f,y=y, iters=1000)
