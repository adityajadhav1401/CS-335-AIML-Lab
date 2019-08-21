import numpy as np
from utils import *


def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	[N, D] = X.shape
	output = np.ones((N,1))
	for i in range(1,D):
		col = X[:,i]
		# only a single colunm added
		if (not isinstance(col[0],str)):
			norm_col = (col - np.mean(col))/(np.sqrt(np.var(col)))
			norm_col = np.reshape(norm_col,(norm_col.shape[0],1))
			output = np.append(output,norm_col,axis=1)
		# multiple colunms added
		else:
			encoding_mat = one_hot_encode(col,np.unique(col))
			output = np.append(output,encoding_mat,axis=1)
	return output.astype(float), Y.astype(float)

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return (-2*np.transpose(np.dot(np.transpose(Y - np.dot(X,W)),X))) + (2*_lambda*W);	

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	[N,D] = X.shape
	W = np.random.randn(D,1)
	for i in range(max_iter):
		# print(i)
		grad_weight = grad_ridge(W,X,Y,_lambda)
		W = W - lr * (grad_weight)
		# print(np.sqrt(np.sum(grad_weight**2)))
		if (np.sqrt(np.sum(grad_weight**2)) < epsilon):
			# print("Iterations : " + str(i))
			break
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	[N,D] = X.shape
	split_size = N // k 
	errors = np.zeros((len(lambdas),k))
	for l in range(len(lambdas)):
		# print(l)
		for i in range(k):
			validation_X = X[((i)*split_size):((i+1)*split_size),:]
			validation_Y = Y[((i)*split_size):((i+1)*split_size),:]
			train_X 	 = np.append(X[0:max(((i)*split_size),0),:],X[min(((i+1)*split_size),N):N,:],axis=0)
			train_Y 	 = np.append(Y[0:max(((i)*split_size),0),:],Y[min(((i+1)*split_size),N):N,:],axis=0)
			_lambda = lambdas[l]
			W = algo(train_X,train_Y,_lambda)		
			errors[l,i] = sse(validation_X,validation_Y,W)	
	
	avg_errors = np.average(errors,axis=1)
	return avg_errors

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	[N,D] = X.shape
	W = np.random.randn(D,1)
	# np.set_printoptions(threshold=np.inf)
	# print(np.matmul(X,W))
	z = np.sum(X**2,axis=0) + 1e-6
	XY = np.dot(X.T,Y)
	XtX = np.dot(X.T,X)
	
	for i in range(max_iter):
		# print(i)
		W_copy = np.copy(W)
		for k in range(D):
			XkY = XY[k]
			XTX_k = XtX[:,k]
			W_k_copy = W[k]
			W[k] = 0
			rho_k = np.sum(XkY - np.dot(XTX_k,W))
			W[k] = W_k_copy		
			# rho_k = np.sum(np.dot(X[:,k].T, (Y - (np.matmul(X,W) - np.matmul(X[:,k:k+1],W[k:k+1,0:1])))))
			# rho_k = np.sum(np.dot(X[:,k].T, (Y - np.matmul((np.append(X[:,0:k],X[:,k+1:D],axis=1)), np.append(W[0:k,0],W[k+1:D,0],axis=0)))))
			z_k = z[k]
			# Case 1 : Assuming W[k] > 0
			w_k1 = (2 * rho_k - _lambda) / (2 * z_k)
			# Case 2 : Assuming W[k] < 0
			w_k2 = (2 * rho_k + _lambda) / (2 * z_k)

			# if (z_k == 1e-6): 
			# 	W[k] = 0
			# 	continue

			# print(z_k,rho_k)
			if (w_k1 >= 0):
				if (w_k2 < 0):
					W[k] = w_k1
					loss_1 = lasso_objective(X,Y,W,_lambda)
					W[k] = w_k2
					loss_2 = lasso_objective(X,Y,W,_lambda)
					if (loss_1 < loss_2):
						W[k] = w_k1
					else:
						W[k] = w_k2
				else:
					W[k] = w_k1
			elif (w_k2 < 0):
				W[k] = w_k2
			else:
				W[k] = 0
		# print(np.sqrt(np.sum((W_copy - W)**2)))
		if (np.sqrt(np.sum((W_copy - W)**2)) < 100):
			# print("Iterations : " + str(i))
			break
	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = list(range(341000,343500,500)) # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	W = ridge_grad_descent(testX,testY,12)
	print(sse(testX,testY,W))
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	# plot_kfold(lambdas, scores)