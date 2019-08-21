import sys

def encoder(filename,p):
	# load and parse maze file
	f = open(filename)
	data = f.readlines()
	width = len(data[0].split())
	height = len(data)
	numStates = width * height
	numActions = 4
	states = []
	for row in data: 
		for state in row.split(): states.append(int(state.strip()))
	start = states.index(2)
	end = states.index(3)
	discountFactor = 1
	p = float(p)

	# create output
	print("numStates", numStates)
	print("numActions", numActions)
	print("start", start)
	print("end", end)
	for i in range(numStates):
		if (states[i] == 1 or states[i] == 3):
			continue
		else:
			north   = i - width
			south   = i + width
			east    = i + 1
			west    = i - 1
			moves   = [north, south, east, west]
			rewards = []
			for move in moves:
				if (move == end): rewards.append(1000)
				else: rewards.append(-1) 
			# find valid moves
			validMoves = [0] * numActions 
			if (states[north] != 1): validMoves[0] = 1
			if (states[south] != 1): validMoves[1] = 1
			if (states[east] != 1): validMoves[2] = 1
			if (states[west] != 1): validMoves[3] = 1
			correctMoveProb = p + (1 - p) / sum(validMoves)
			randomMoveProb = (1.0 - p) / sum(validMoves)

			# print transitions for valid moves
			# North -> 0
			# South -> 1
			# East -> 2
			# Wwst -> 3
			if (states[north] != 1): 
				print("transitions", i, "0", north, rewards[0], correctMoveProb)
				for j in range(numActions):
					if ((validMoves[j] == 1) and (j != 0) and (randomMoveProb != 0)):
						print("transitions", i, "0", moves[j], rewards[j], randomMoveProb)
			if (states[south] != 1): 
				print("transitions", i, "1", south, rewards[1], correctMoveProb)
				for j in range(numActions):
					if ((validMoves[j] == 1) and (j != 1) and (randomMoveProb != 0)):
						print("transitions", i, "1", moves[j], rewards[j], randomMoveProb)
			if (states[east] != 1): 
				print("transitions", i, "2", east, rewards[2], correctMoveProb)
				for j in range(numActions):
					if ((validMoves[j] == 1) and (j != 2) and (randomMoveProb != 0)):
						print("transitions", i, "2", moves[j], rewards[j], randomMoveProb)
			if (states[west] != 1): 
				print("transitions", i, "3", west, rewards[3], correctMoveProb)
				for j in range(numActions):
					if ((validMoves[j] == 1) and (j != 3) and (randomMoveProb != 0)):
						print("transitions", i, "3", moves[j], rewards[j], randomMoveProb)

	print("discount ", discountFactor)

encoder(sys.argv[1],sys.argv[2])