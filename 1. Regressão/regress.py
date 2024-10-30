import numpy as np
import matplotlib.pyplot as plt

def loadData(filepath='0b. DADOS/aerogerador.dat', checkShape=True):
  data = np.loadtxt(filepath)
  if(checkShape):
    print(f"Data shape from '{filepath}': {data.shape}")
  return data

y = (2250,1)

data = loadData(checkShape=False)
x1 = data[:,0:1]
x2 = data[:,1:]

x1.shape = (2250,1)
x2.shape = (2250,1)

plt.scatter(x1,x2)

plt.show()