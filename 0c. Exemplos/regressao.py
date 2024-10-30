import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([
480   ,
500   ,
380   ,
1100  ,
1100  ,
230   ,
490   ,
250   ,
300   ,
510   ])

y = np.array([
180,
150,
170,
350,
460,
60,
240,
90,
110,
250])



x1.shape = (10,1)
y.shape = (10,1)

plt.scatter(x1,y,color='purple')
plt.xlim(200,1200)

X1 = np.concatenate((np.ones((10,1)),x1),axis=1)

b_hat = np.linalg.inv(X1.T@X1)@X1.T@y
# b_hat1 = np.linalg.pinv(X1.T@X1)@X1.T@y
# b_hat2 = np.linalg.lstsq(X1,y)[0]

x_axis = np.linspace(0,1300,1300).reshape(1300,1)

X_axis = np.concatenate((np.ones((1300,1)),x_axis),axis=1)

y_hat = X_axis@b_hat

plt.plot(x_axis,y_hat,color='blue')

plt.show()




