import numpy as np
import matplotlib.pyplot as plt

def f(mu, sigma, x): #Função de Probabilidade de Densidade p/ Distribuição Normal
  return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))


x_normal = np.random.normal(loc=0, scale=1, size=(100,))
# plt.hist(x_normal, bins=75)
mu, sigma, n = 0,1,3

x_axis = np.linspace(-5, 5, 5000)
x1 = np.linspace(mu-n*sigma, mu+n*sigma)
plt.fill_between(x1, f(mu, sigma,x1), alpha=0.3)
plt.xlim(-4.5,4.5)
plt.plot(x_axis, f(mu, sigma, x_axis))



plt.show()