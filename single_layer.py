import numpy as np
import plotfunc

class Classifer(object):
    def __init__(self):
        self.weight = np.random.random()
        self.bias = np.random.random()
    def SetPar(self,w,b):
        self.weight = w
        self.bias = b
    def LearningQ(self,x,y,epochs,lr):
        z1 = []
        z2 = []
        for j in xrange(epochs):
            delta_y = 0.0
            cost = 0.0
            #for px,py in zip(x,y):
            z = x*self.weight+self.bias
            a = sigmod(z)
            cost += (y-a)*(y-a)/2
            delta_y += a*a*(1-a)
            self.weight = self.weight - lr*delta_y
            self.bias = self.bias - lr*delta_y
            z1.append(j)
            z2.append(cost)
        plotfunc.plotfun(z1,z2,np.max(z2)+0.1)
    def LearningCE(self,x,y,epochs,lr):
        z1 = []
        z2 = []
        for j in xrange(epochs):
            delta_w = 0.0
            delta_b = 0.0
            cost = 0.0
            z = x*self.weight+self.bias
            a = sigmod(z)
            cost += -(y*np.log(a)+(1-y)*np.log(1-a))
            #cost += (y-a)*(y-a)/2
            delta_w += x*(a-y)
            delta_b += a-y
            self.weight = self.weight - delta_w*lr
            self.bias = self.bias - delta_b*lr
            z1.append(j)
            z2.append(cost)
        plotfunc.plotfun(z1,z2,np.max(z2)+0.1)
def sigmod(z):
    return 1.0/(1.0+np.exp(-z))