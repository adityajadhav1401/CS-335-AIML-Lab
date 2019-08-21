import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
	"""
	Bagging classifier.

	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	
	"""

	def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

		self.ratio = ratio
		self.num_classifiers = num_classifiers
		self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]
		self.legalLabels = legalLabels

	def train( self, trainingData, trainingLabels):
		"""
		The training loop samples from the data "num_classifiers" time. Size of each sample is
		specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
		"""

		self.features = trainingData[0].keys()
		"*** YOUR CODE HERE ***"
		sample_size = int(len(trainingData) * self.ratio)
		train_data_size = len(trainingData)
		index = 1
		for classifier in self.classifiers:
			sample_index = np.random.choice(train_data_size,sample_size,replace=True)
			sample_data = [trainingData[i] for i in sample_index]
			sample_labels = [trainingLabels[i] for i in sample_index]
			print("Training Classifier " + str(index))
			classifier.train(sample_data,sample_labels,sample_weights=None)
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
		guesses = []
		for datum in data:
			votes = util.Counter()

			for l in self.legalLabels:
				votes[l] = 0

			for classifier in self.classifiers:
				votes[classifier.classify([datum])[0]] += 1

			guesses.append(votes.argMax())
		
		return guesses
		# util.raiseNotDefined()
