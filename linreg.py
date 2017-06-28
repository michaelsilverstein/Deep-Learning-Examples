import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import itertools

#Deep Learning
#http://www.deeplearningbook.org/
#Figure 5.1
#Linear regression

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"] #For Latex later on

def datagen(dim,size,noise=0.3):
    #Input:
    #|dim = number of features
    #|size = sample size
    #|noise = jitter parameter
    #Output:
    #x: features in dimensions `dim` of size `size`
    #y: observations of size `size
    jitter = noise #Spread parameter of noise
    seed = np.array([np.random.rand(size)]*dim) #Linear relationship in dim dimensions
    noise = np.random.rand(dim,size) * jitter
    x = np.vstack([seed+noise,np.ones(size)]).T #append column of ones for bias (y-intercept)
    y = seed[0] #Vector of observations
    return x,y

def MSE(y_,y,size):
    #Compute MSE for a vector of predictions y_ and observations y
    return 1./size*np.power(np.linalg.norm(y_-y),2)


def linreg(x, y):
    # Train model parameters, w, based on matrix of features,x, and observations, y
    w = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(x.T, x)),
            x.T),
        y)
    return w


def gridsearch(x,y,size, ax=False, amin=-.5, amax=1, contour=False):
    # Find optimal parameters through a grid search with resolution sizeXsize
    # Input:
    # |x: feature matrix
    # |y: observation vector
    # |size: step size between points in grid; grid resolution
    # |ax: When plot is desired, pass ax object to plot
    # |amin,amax = axis min and max
    # |contour: plot contour plot instead of scatter
    # Output:
    # |w: parameters that minimize MSE

    # Generate mesh
    seed = np.linspace(amin, amax, size)
    slope, b = np.meshgrid(seed, seed)  # 50x50 grid from 0 to 2
    combos = np.vstack([slope.ravel(), b.ravel()])  # All pairwise combinations of parameters in grid

    # Compute y_hat for each set of parameters
    y_tests = np.dot(x, combos).T  # y_tests = [y_hats{1},...,y_hats{size}], for pairs of parameters 1,...,size
    MSEs = np.array([MSE(y_, y) for y_ in y_tests]).reshape(size, size)  # Calculate MSE for each set of y_hats
    minval_ix = np.unravel_index(np.argmin(MSEs), (size, size))  # Find index at which MSE is the smallest

    w = [seed[minval_ix[1]], seed[minval_ix[0]]]

    if ax:
        if contour:
            ax.contourf(slope, b, MSEs)
        else:
            ax.scatter(slope, b, c=MSEs)
        ax.set_xlabel('slope')
        ax.set_ylabel('b')
    return w