import numpy as np
import matplotlib.pyplot as plt

#Deep Learning
#http://www.deeplearningbook.org/
#Figure 5.1
#Linear regression

# y_hat = w.T x + b
# w = (X.T X)^-1 X.T y
def MSE(y_,y):
    #Mean square error
    return 1./len(y)*np.power(np.linalg.norm(y_-y),2)
def plots():
    ##Plot
    fig, axs = plt.subplots(ncols=2)

    # Linear regression
    axs[0].plot(x[:, 0], y, 'o', label='Actual')
    axs[0].plot(x[:, 0], y_hat, label='Predicted')
    axs[0].legend()
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_ylabel(r'$y$')

    # Comparing ws
    axs[1].plot(ws[:,0], MSEs, 'black')
    axs[1].plot(w[0], MSE_, 'o')
    axs[1].set_xlabel(r'$w_1$')
    axs[1].set_ylabel(r'$MSE^{(train)}$')

    fig.suptitle('Figure 5.1')
    plt.show()
    # plt.savefig('figures/fig5.1.png')
"""MAIN"""
size = 100
dim=1
jitter = 0.3 #Spread parameter
x = np.array([np.concatenate([r,[1]]) for r in np.random.rand(size,dim)])
y = x[:,0]+(np.random.rand(size)*jitter) #jittery but linearish

##Calculate best w that minimizes MSE
w = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(x.T,x)),
                    x.T),
                        y)
y_hat = np.dot(x,w)#Dot product

#Explore MSE landscape for w1 for 1 feature case
if dim == 1:
    ws = np.array([[x_,w[1]] for x_ in np.linspace(0,w[0]+1,20)]) #From 0 to a bit past the solution, keeping y-int constant
    y_hats = np.array([np.dot(x,w_) for w_ in ws]) #Predicted y for each w
    MSEs = [MSE(y_,y) for y_ in y_hats] #MSE for all ws
    MSE_ = MSE(y_hat,y) #For the best w, calculated in the prev step

plots()