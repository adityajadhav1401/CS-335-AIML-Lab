import sys

def mdp(filename):
	# load and parse MDP file
	f = open(filename)
	data = f.readlines()
	numStates = int(data[0].split()[1])
	numActions = int(data[1].split()[1])
	start = int(data[2].split()[1])
	end = [int(i) for i in data[3].split()[1:]]
	transitions = {}
	for i in range(4,len(data)-1):
		val = data[i].split()
		s_curr 	= int(val[1])
		action 	= int(val[2])
		s_prime	= int(val[3])
		reward 	= float(val[4])
		prob	= float(val[5])
		try: transitions[s_curr][action].append((s_prime, reward, prob))
		except:
			try: transitions[s_curr][action] = [(s_prime, reward, prob)]
			except: transitions[s_curr] = {action: [(s_prime, reward, prob)]}
	discountFactor = float(data[len(data)-1].split()[1])
	values = [0.0] * numStates
	policy = [-1] * numStates

	
	# perform value iteration 
	new_values = [0.0] * numStates
	t = 0
	while True:
		# print(t)
		for s in transitions:
			if s not in end:
				max_score = -float('inf')
				max_action = -1
				for a in transitions[s]:
					# calculate score
					score = 0
					for transition in transitions[s][a]: 
						score += transition[2]*(transition[1] + discountFactor*values[transition[0]])
			        # update max scores
					if score > max_score:
						max_score = score
						max_action = a
				new_values[s] = max_score
				policy[s] = max_action
		
		t += 1
		stop = True
		for s in transitions:
			if (abs(new_values[s] - values[s]) > 1e-16):
				stop = False
				break
		# update values		
		for s in transitions:
			values[s] = new_values[s]
		# stop if not much has changed
		if stop:
			break

	for i in range(numStates):
		print(round(values[i], 11), policy[i])
	print("iterations", t)

mdp(sys.argv[1])