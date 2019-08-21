import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(classifier, data):
	return classifier.classify(data)

class AdaBoostClassifier:
	"""
	AdaBoost classifier.

	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	
	"""

	def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
		self.legalLabels = legalLabels
		self.boosting_iterations = boosting_iterations
		self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
		self.alphas = [0.0]*self.boosting_iterations

	def train( self, trainingData, trainingLabels):
		"""
		The training loop trains weak learners with weights sequentially. 
		The self.classifiers are updated in each iteration and also the self.alphas 
		"""
		
		self.features = trainingData[0].keys()
		"*** YOUR CODE HERE ***"
		train_data_size = len(trainingData)
		num_classifiers = len(self.classifiers)
		weights = np.array([1.0/(train_data_size) for _ in range(train_data_size)])
		index = 1
		for k in range(num_classifiers):
			classifier = self.classifiers[k]
			print("Training Classifier " + str(index))

			classifier.train(trainingData,trainingLabels,weights)

			error = 0.0
			pred = classifier.classify(trainingData)
			for i in range(train_data_size):
				if (pred[i] != trainingLabels[i]):
					error = error + weights[i]
			print("Error " + str(error))
			for i in range(train_data_size):
				if (pred[i] == trainingLabels[i]):
						weights[i] = weights[i] * (error) / (1 - error)
				# else:
				# 	weights[i] = weights[i] * (1 - error) / (error)  

			self.alphas[k] = np.log((1 - error)/(error))
			print("Alpha " + str(self.alphas[k]))
			weights = weights / (np.sum(weights))
			index += 1


		# util.raiseNotDefined()

	def classify( self, data):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label. This is done by taking a polling over the weak classifiers already trained.
		See the assignment description for details.

		Recall that a datum is a util.counter.

		The function should return a list of labels where each label should be one of legaLabels.
		"""

		"*** YOUR CODE HERE ***"
		guesses = np.zeros(len(data))

		for k in range(len(self.classifiers)):
			classifier = self.classifiers[k]
			guesses += np.dot(classifier.classify(data),self.alphas[k])
		
		guesses = np.sign(guesses)
		guesses[np.where(guesses == 0)[0]] = np.repeat(np.expand_dims(np.random.choice([-1,1]),axis=0),len(np.where(guesses == 0)[0]),axis=0)
		return guesses
		# util.raiseNotDefined()