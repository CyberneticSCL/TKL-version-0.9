import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
import scipy.io
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from PMKL import KernelFunctions
from PMKL import Optimization
from PMKL import Transformation
import time
Kernel = KernelFunctions.Kernel
makeK  = KernelFunctions.makeK
TKtest = KernelFunctions.TKtest

findP     = Optimization.findP
findAlpha = Optimization.findAlpha
monomials = Transformation.monomials

class PMKL():
    def __init__(self, C = 10,  degree = 1, bound = 0.1, epsilon = 0.1, maxit = 100, tol = 1.e-6, probability = False, to_print= True):
        '''
        C:      Regularization parameter, smaller values lead to mappings that are more general.
        
        degree:  Degree of the TK Kernel function (delta parameter, gamma = 0)
        
        bound:   Area of integration is [-bound,1+bound]
        
        epsilon: Epsilon parameter for regression problems
        
        maxit:   Maximum number of iterations.
        
        tol:     Tolerance of the TKL optimization algorithm
        
        probability: This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation
        
        to_print:  
        '''
#         if params   == None:
        params = ParamsTK(degree, bound, epsilon, maxit, tol)
            
        self.Params = params
        self.to_print = to_print
        self.prob = probability
        self.Params.C = C

        #### PARAMETERS #####
        #### OPT ####
        
        self.Opt = Opt()


        
    def fit(self, x, y, subset_Z = 2):
        '''
        x:      Inputs to be mapped to outputs y. It should be numpy array (n_samples, n_features)
        y:      Outputs. (n_samples)
        subset_Z: if we need to include K22 or only K11 term
        '''
        start = time.time()
        Eps = self.Params.epsilon
        tol = self.Params.tol
        maxit = self.Params.maxit
        if len(np.unique(y)) == 2:
            TKL_type = 'Classification';
        else:
            TKL_type = 'Regression';
        self.Type = TKL_type        
        if len(y.shape) < 2:
            y = y[:, np.newaxis]
            
        if y.shape[0]>1 & y.shape[1] == 1 : 
            self.y = y 
        elif y.shape[1]>1 & y.shape[0] == 1  :
            self.y = y.T
        else:
            error('y must be a vector')
            
        if len(self.y) == x.shape[1] : 
            self.xOld = x.T
        elif len(self.y) == x.shape[0]:
            self.xOld = x
        else:
            error('x and y must contain the same number of points')
        
        
        scaleFactor = MinMaxScaler()
        self.x  = scaleFactor.fit_transform(self.xOld)
        self.scaleFactor = scaleFactor
        
        num, dim = self.x.shape # Dimension and number of inputs
        
        self.Params.Lower =  self.x.min(axis = 0) - self.Params.bound # Lower bounds of integration
        self.Params.Upper =  self.x.max(axis = 0) + self.Params.bound # Upper bounds of integration
        
        self.Kernel = Kernel(self.x, self.Params.Lower, self.Params.Upper, self.Params.degree)
        Kernel.subset_Z = subset_Z;
        self.Params.q = 2 * self.Kernel.Z.shape[1]
#         print(self.Kernel.K[1,1].shape, self.Kernel.Z.shape)
        q = self.Params.q 
        self.Params.P = np.eye(q) # Initialize P matrix

        
        if self.Type == 'Classification':
            self.model = SVC(C = self.Params.C, kernel = 'precomputed', probability = self.prob)
        else:
            self.model = SVR(C = self.Params.C, epsilon = self.Params.epsilon, kernel = 'precomputed')
            
        maxit = self.Params.maxit
        go = True
        iteration = 0
        Obj = findAlpha(self, self.Kernel)
        self.Opt.Obj.append(Obj)
        self.Opt.diff = np.abs(self.Opt.Obj[-1]-self.Opt.Obj[-2])/np.abs(self.Opt.Obj[-1] + self.Opt.Obj[-2])*200 # Calculates the percent difference between objective values
        end = time.time()
        self.parsing_time = end - start
        if self.to_print:
            print('Iteration   |  Objective   |       Dual Gap      | \n')
            print('------------+--------------+---------------------| \n')
#         return self
        while go:
            
#             go = False
#             break
            iteration = iteration+1;
            findP(self, self.Kernel) # Updates the P matrix which parameterizes the Positive Matrix Kernel Function.
            if (iteration > maxit) or (np.min([self.Opt.dualGap2[-1],self.Opt.dualGap[-1]]) < self.Params.tol) or (self.Opt.StepLength[-1] < 1/100*self.Params.tol) :
            ### Quits if maxit > current iteration or if the objective function change was too small.
                go = False;
            if self.to_print:
                print('%10d  |  %1.4e  |  %1.4e \n' % (iteration, self.Opt.Obj[-1], np.min(self.Opt.dualGap[-1])))
#         Ktrain = makeK(self.Kernel, self.Params.P)
#         self.model = self.model.fit(Ktrain, self.y)
        return self
        
    def predict(self, Xtest_old):
        '''
        
        '''
        xTest = self.scaleFactor.transform(Xtest_old)
        xtrain = self.x[self.Params.pos, :]
        
        dimx = xtrain.shape[1]
        numx = xtrain.shape[0]
        numtest = xTest.shape[0]
        
        Kt = TKtest(xtrain, xTest, 
                    self.Kernel.Z[self.Params.pos, :], monomials(xTest,self.Params.degree),
                    self.Params.Lower,self.Params.Upper, self.Params.P)
        
        
        if self.Type == 'Classification':
            vec = self.Params.alpha*self.y[self.Params.pos]
            yPred = vec.T@Kt - self.Params.rho# + self.model.intercept_
#             yPred = np.sign(yPred)

        else:
            yPred = self.Params.alpha.T@Kt - self.Params.rho# + self.model.intercept_
            
        if yPred.shape[0] == 1:
            yPred = yPred[0, :]
        return yPred
    
    def predict_proba(self, Xtest_old):
        '''
        Compute probabilities of possible outcomes for samples in X.
        The model need to have probability information computed at training time: fit with attribute probability set to True.
        
        input:
        X array-like of shape (n_samples, n_features)
        
        output:
        
        '''
        xTest = self.scaleFactor.transform(Xtest_old)
        xtrain = self.x[self.model.support_, :]
        
        dimx = xtrain.shape[1]
        numx = xtrain.shape[0]
        numtest = xTest.shape[0]
        
        Kt = TKtest(xtrain, xTest, 
                    self.Kernel.Z[self.model.support_, :], monomials(xTest,self.Params.degree),
                    self.Params.Lower,self.Params.Upper, self.Params.P)
        yPred = self.model.predict_proba(Kt)
        
#         if self.Type == 'Classification':
#             yPred = np.sign(yPred)
            
        if yPred.shape[0] == 1:
            yPred = yPred[0, :]
        return yPred
        
class ParamsTK():
    '''
    [params] = paramsTK(degree,bound,epsilon,maxit,tol) function takes
    degree, bound, epsilon, maxit, and tol parameters as inputs for TK kernel
    functions.  Entering an empty array as an input means the default value
    will be used.
    
    INPUT
    degree:  Degree of the TK Kernel function (delta parameter, gamma = 0)
    bound:   Area of integration is [-bound,1+bound]
    epsilon: Epsilon parameter for regression problems
    maxit:   Maximum number of iterations.
    tol:     Tolerance of the TKL optimization algorithm.
    
    OUTPUT
    params: A variable containing the saved parameter values to be given to TKL.  
    '''
    def __init__(self, degree = 1, bound = 0.1, epsilon = 0.1, maxit = 100, tol = 1.e-6):
        self.degree = degree
        self.bound = bound
        self.epsilon = epsilon
        self.maxit = maxit
        self.tol = tol

        self.kernel = 'TK' 
        
        
class Opt():
    def __init__(self):
        self.StepLength = []
        self.dualGap = []
        self.dualGap2 = [np.inf]
        self.l  = [-np.inf  ]
        self.Obj = [-np.inf ]
        self.StepLength.append(1)
        self.TT = [] 
        
        
def loadex1():
    mat = scipy.io.loadmat('PMKLpy/CircleData.mat')['Data']
    X = mat[0,0][0]
    y = mat[0,0][1]
    return X, y


        