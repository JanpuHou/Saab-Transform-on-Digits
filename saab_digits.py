# v2020.03.25

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import time

class Saab():
    def __init__(self, num_kernels=-1, useDC=True, needBias=True):
        self.par = None
        self.Kernels = []
        self.Bias = []
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.needBias = needBias
        self.trained = False

    def remove_mean(self, X, axis):
        feature_mean = np.mean(X, axis=axis, keepdims=True)
        X = X - feature_mean
        return X, feature_mean

    def fit(self, X): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')
        X, self.Mean0 = self.remove_mean(X.copy(), axis=0)
        X, dc = self.remove_mean(X.copy(), axis=1)
        self.Bias = np.max(np.linalg.norm(X, axis=1)) * 1 / np.sqrt(X.shape[1])
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        # pca = PCA(n_components=self.num_kernels, svd_solver='full').fit(X)
        pca = IncrementalPCA(n_components=self.num_kernels).fit(X)
        kernels = pca.components_
        energy = pca.explained_variance_ / np.sum(pca.explained_variance_)
        if self.useDC == True:  
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))     
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1])) / np.sqrt(largest_ev)
            kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
            energy = np.concatenate((np.array([largest_ev]), pca.explained_variance_[:-1]), axis=0)
            energy = energy / np.sum(energy)
        self.Kernels, self.Energy = kernels, energy
        self.trained = True
        
    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        X -= self.Mean0
        if self.needBias == True:
            X += self.Bias
        dc = np.mean(X.copy(), axis=1, keepdims=True)
        X = np.matmul(X, np.transpose(self.Kernels))
        if self.needBias == True and self.useDC == True:
            X[:, 0] -= self.Bias
        return X, dc
    
    def inverse_transform(self, X, DC):
        assert (self.trained == True), "Must call fit first!"
        assert (DC.shape[0] == X.shape[0]), "Input shape not match! 'X' and 'DC'"
        X = X.astype('float32')
        DC = DC.astype('float32')
        if self.needBias == True and self.useDC == True:
            X[:, 0] += self.Bias
        X = np.matmul(X, self.Kernels)
        if self.needBias == True:
            X -= self.Bias 
        #X += DC
        X += self.Mean0
        return X

if __name__ == "__main__":
    from sklearn import datasets
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(data.shape))    
    print(" ")
    print(" ")
    import matplotlib.pyplot as plt
    print('1st case:, \n')
    for i in range(36):
      plt.subplot(6, 6, i + 1)
      plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()    
       
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:1000]
    saab = Saab(num_kernels=-1, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    print('2nd case:, \n')    
    for i in range(36):
     plt.suptitle('number of kernels = 64')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()



    saab = Saab(num_kernels=32, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    
    print('3rd case:, \n')
    for i in range(36):
     plt.suptitle('number of kernels = 32')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()   


    saab = Saab(num_kernels=16, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    
    print('3rd case:, \n')
    for i in range(36):
     plt.suptitle('number of kernels = 16')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

    saab = Saab(num_kernels=8, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    
    print('3rd case:, \n')
    for i in range(36):
     plt.suptitle('number of kernels = 8')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

    saab = Saab(num_kernels=4, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    
    print('3rd case:, \n')
    for i in range(36):
     plt.suptitle('number of kernels = 4')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()  

    saab = Saab(num_kernels=2, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    print(Xt.shape)
    Y = saab.inverse_transform(Xt, dc)
    
    print('3rd case:, \n')
    for i in range(36):
     plt.suptitle('number of kernels = 2')
     plt.subplot(6, 6, i + 1)
     y=Y[i,].reshape(-1,8)
     plt.imshow(y, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()       