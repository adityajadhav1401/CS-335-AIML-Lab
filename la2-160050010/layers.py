import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.dot(X, self.weights) + np.kron(np.ones((n, 1)), self.biases)
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# for i in range(n):
		# 	temp = (delta[i,:] * derivative_sigmoid(self.data[i,:]))
		# 	temp = np.reshape(temp,(temp.shape[0],-1))
		# 	new_delta[i,:] = np.matmul(self.weights,temp).T
		# 	grad_weight[i,:,:] = np.matmul(temp, np.reshape(activation_prev[i,:],(self.in_nodes,-1)).T).T
		# 	grad_bias[i,:] = temp.T

		temp = (delta * derivative_sigmoid(self.data))
		new_delta = np.matmul(self.weights,temp.T).T
		grad_weight = np.matmul(activation_prev.T,temp)
		grad_bias = temp
		self.weights -= lr * grad_weight
		self.biases -= lr * np.sum(grad_bias,axis=0)  
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		# for i in range(n):
		# for j in range(self.out_depth):
		for x in range(self.out_row):
			for y in range(self.out_col):
				# self.data[i,j,x,y] = np.sum(X[i,:,(x*self.stride):(x*self.stride+self.filter_row),(y*self.stride):(y*self.stride+self.filter_col)] * \
				# 	self.weights[j,:,:,:]) + self.biases[j]
				self.data[:,:,x,y] = np.sum(np.repeat(np.expand_dims(X[:,:,(x*self.stride):(x*self.stride+self.filter_row),(y*self.stride):(y*self.stride+self.filter_col)],axis=1),self.out_depth,axis=1) * \
					self.weights,axis=(2,3,4)) + np.repeat(np.expand_dims(self.biases,axis=0),n,axis=0)
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		grad_weight = np.zeros((n,self.out_depth, self.in_depth, self.filter_row, self.filter_col))
		temp = (delta * derivative_sigmoid(self.data))

		# for i in range(n):
		# for j in range(self.out_depth):
		for x in range(self.filter_row):
			for y in range(self.filter_col):			
				# grad_weight[i,j,:,x,y] =  np.sum(activation_prev[i,:,x:(x+self.stride*self.out_row):self.stride,y:(y+self.stride*self.out_col):self.stride] * \
				# 							np.kron(np.ones((self.in_depth,1,1)),temp[i,j,:,:]),axis=(1,2))
				grad_weight[:,:,:,x,y] =  np.sum(np.repeat(np.expand_dims(activation_prev[:,:,x:(x+self.stride*self.out_row):self.stride,y:(y+self.stride*self.out_col):self.stride],axis=1),self.out_depth,axis=1) * \
											np.repeat(np.expand_dims(temp,axis=2),self.in_depth,axis=2),axis=(3,4))
		grad_bias = np.sum(temp,axis=(0,2,3))	

		new_delta = np.zeros((n,activation_prev.shape[1],activation_prev.shape[2],activation_prev.shape[3]))
		# for i in range(n):
		# for j in range(self.in_depth):
		for x in range(self.in_row):
			for y in range(self.in_col):	
				xf = max(int(np.ceil((x+1-self.filter_row)/self.stride)),0)
				yf = max(int(np.ceil((y+1-self.filter_row)/self.stride)),0)
				xl = min(x//self.stride+1,self.out_row)
				yl = min(y//self.stride+1,self.out_row)
				# new_delta[i,j,x,y] += np.dot(temp[i,:,k,l], np.sum(self.weights[:,:,x-k*self.stride,y-l*self.stride],axis=1))
				new_delta[:,:,x,y] += np.sum(np.multiply(np.repeat(np.expand_dims(temp,axis=2),self.in_depth,axis=2)[:,:,:,xf:xl,yf:yl], \
					np.flip(np.repeat(np.expand_dims(self.weights,axis=0),n,axis=0)[:,:,:,max(0,x-((xl-1)*self.stride)):max(0,x-((xf-1)*self.stride)):self.stride, max(0,y-((yl-1)*self.stride)):max(0,y-((yf-1)*self.stride)):self.stride], (3,4))),\
						axis=(1,3,4))


		self.weights -= lr * np.sum(grad_weight,axis=0)
		self.biases -= lr * np.sum(grad_bias,axis=0)  
		return new_delta
		# raise NotImplementedError

		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		# for i in range(n):
		# for j in range(self.out_depth):
		for x in range(self.out_row):
			for y in range(self.out_col):
				self.data[:,:,x,y] = np.mean(X[:,:,(x*self.stride):(x*self.stride+self.filter_row),(y*self.stride):(y*self.stride+self.filter_col)],axis=(2,3))
				# self.data[:,j,x,y] = np.mean(X[:,j,(x*self.stride):(x*self.stride+self.filter_row),(y*self.stride):(y*self.stride+self.filter_col)],axis=(1,2))
		return self.data		
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		new_delta = np.zeros((n,activation_prev.shape[1],activation_prev.shape[2],activation_prev.shape[3]))
		# for i in range(n):
		for x in range(self.in_row):
			for y in range(self.in_col):	
				new_delta[:,:,x,y] += (delta[:,:,x//self.stride,y//self.stride]/(self.stride**2))
		return new_delta
		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))