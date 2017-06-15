import numpy as np
import matplotlib.pyplot as plt

#Deep Learning
#http://www.deeplearningbook.org/
#Chapter 5 Page 106
#Linear regression

# y_pred = w.T x + b
# w = (X.T X)^-1 X.T y

size = 50

x = np.array([[r,1] for r in np.random.rand(size)])
y = x[:,0]+(np.random.rand(size)*.3) #jittery but linearish

w = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(x.T,x)),
                    x.T),
                        y)
y_hat = w.T * x
y_pred = y_hat[:,0] + y_hat[:,1]

plt.plot(x[:,0],y,'o',label='Actual')
plt.plot(x[:,0],y_pred,label='Predicted')
plt.legend()
plt.show()