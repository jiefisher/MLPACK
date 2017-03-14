import numpy as np
import math
train_X=np.array([[1],[2],[4]])
y=np.mat([2,3,5])
EX=np.mean(train_X,axis=0)
VX=np.var(train_X,axis=0)

def poly_basis_function(x,j):
	f=x**j
	return f
def linear_regression(train_X,y,n_feature="single",basis_func="Gaussian",b_iter=1,method="least_sqare"):
	if method=="least_sqare":#To error function,use least_square to minimize the error function
		if n_feature=="single":# if the number of features is 1
			
			if basis_func=="Poly":#use polynomial for basis function
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
linear_regression(train_X,y,n_feature="single",basis_func="Poly",b_iter=3,method="least_sqare")