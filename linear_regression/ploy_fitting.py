#########
#author:jiefisher
#########
import numpy as np
import math
###
#train data:
#train_X is the feature of data
#y is the label of data
###
train_X=np.array([[1],[2],[4]])
y=np.mat([2,4,8])

###
#the basis function of regression
#ploy is use polynomial as the basis function,shown as:
#y=w0+w1*x+w2 * x^2+...+wn * x^n
###
def poly_basis_function(x,j):
	f=x**j
	return f
###
#the ploy_fitting can be used to curve linear fitting
#train_X:the feature of train data
#y:the label of train data
#feature: if the number of features is 1,use "single",else,"multi"
#basis_func:the basis function,like"Poly" as polynomial fitting
#b_iter:the power of poly function,such as x^n
#method:the method of minimizing the error function,such"leaast_square" for minimizing least square,"SGD"for stochastic gradient descent
#epsilon:loss_rate of error function("SGD" specially)
#lambdas:parameter for preventing overfitting
#alpha:learning rate("SGD" specially)
#g_iter:iterations of "SGD"("SGD" specially)
###


def ploy_fitting(train_X,y,n_feature="single",basis_func="Poly",b_iter=1,method="least_sqare",epsilon=.001,lambdas=.03,alpha=.00001,g_iter=10000):
	###
	#the error function is:
	#1/2*(y-f(X,W))^2
	###
	if method=="least_sqare":#use least_square to minimize the error function
	###
	# the formula of least_square is:
	# W=(ΦT*Φ)^-1*ΦT*t
	# where Φ=[Φ0(x1) Φ1(x1)...ΦM(x1)
	# 			:		:		:
	# 		   Φ0(xN) Φ1(xN)...ΦM(xN)
	# ]
	# whereΦj(x)=x^j
	# W is parameter matrix
	###
		if n_feature=="single":# if the number of features is 1
			
			if basis_func=="Poly":#use polynomial in basis function
				ma=[[0 for i in range(b_iter)] for j in range(train_X.shape[0])]
				for i in range(0,train_X.shape[0]):
					for j in range(0,b_iter):
						ma[i][j]=poly_basis_function(train_X[i][0],j)
		if n_feature=="multi":# if the number of features is n
			if basis_func=="Poly":
				ma=[[0 for i in range(train_X.shape[1])] for j in range(train_X.shape[0])]
				for i in range(0,train_X.shape[0]):
					for j in range(0,train_X.shape[1]):
						ma[i][j]=poly_basis_function(train_X[i][j],1)
		mat_X=np.mat(ma)
		mat_middle=mat_X.T*mat_X
		mat_i=mat_middle.I*mat_X.T*y.T
		print(mat_i)#the parameter matrix
	if method=="SGD":#use stochastic gradient descent to minimize the error function
	###
	#the formula of stochastic gradient descent is:
	#[W]=[W]+alpha*(y(i)-f(i)(W,X))*[X]
	#where alpha is the learning rate
	#[W] is parameter matrix
	###
		if n_feature=="single":# if the number of features is 1
			
			if basis_func=="Poly":#use polynomial in basis function
				ma=[[0 for i in range(b_iter)] for j in range(train_X.shape[0])]
				for i in range(0,train_X.shape[0]):
					for j in range(0,b_iter):
						ma[i][j]=poly_basis_function(train_X[i][0],j)
		if n_feature=="multi":# if the number of features is n
			
			if basis_func=="Poly":#use polynomial in basis function
				ma=[[0 for i in range(train_X.shape[1])] for j in range(train_X.shape[0])]
				for i in range(0,train_X.shape[0]):
					for j in range(0,train_X.shape[1]):
						ma[i][j]=poly_basis_function(train_X[i][j],1)

		ma=np.mat(ma)
		loss=1.0
		weights=np.mat(np.random.random(b_iter))
		i=0
		
		#print(n)
		for i in range(0,g_iter) :
			for j in range(0,train_X.shape[0]):
				h=np.sum(ma[j]*weights.T)
				error=y[0,j]-h
				loss=(1/2)*(error**2+lambdas*np.sum(weights*weights.T))
				weights=weights+alpha*(error*ma[j]+lambdas*weights)
				print(loss)
				if np.sum(loss)<epsilon:
						break
			if  np.sum(loss)<epsilon:
				break
		print(weights)#the parameter matrix

ploy_fitting(train_X,y,n_feature="single",basis_func="Poly",b_iter=2,method="least_sqare")

