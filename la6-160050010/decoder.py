import sys
import random
import math
import numpy as np
np.random.seed(0)

def decoder(gridfile,policyfile,p):
    f = open(gridfile)
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
    policy = []
    values = []
    p = float(p)
    f.close()

    f = open(policyfile)
    data = f.readlines()
    for row in data:
        if (row.startswith("iteration")):
            break
        val = row.split()
        values.append(float(val[0]))
        policy.append(int(val[1]))

    state = start
    directions = {0: 'N',1: 'S',2: 'E',3: 'W'}
    while(state != end):
        # print(state)
        north = state - width
        south = state + width
        east = state + 1
        west = state - 1
        moves   = [north, south, east, west]
        # print(moves)
        # find valid moves
        validMoves = [0,0,0,0]
        if (states[north] != 1):
            validMoves[0] = 1
        if (states[south] != 1): 
            validMoves[1] = 1
        if (states[east] != 1): 
            validMoves[2] = 1
        if (states[west] != 1): 
            validMoves[3] = 1

        correctMoveProb = p + (1.0 - p) / sum(validMoves)
        randomMoveProb = (1.0 - p) / sum(validMoves)

        pdfValid = [0.0, 0.0, 0.0, 0.0]
        if (states[north] != 1):
            pdfValid[0] = randomMoveProb 
        if (states[south] != 1): 
            pdfValid[1] = randomMoveProb
        if (states[east] != 1): 
            pdfValid[2] = randomMoveProb
        if (states[west] != 1): 
            pdfValid[3] = randomMoveProb
        
        pdfValid[policy[state]] += p
        cdfValid = [sum(pdfValid[:k+1]) for k in range(numActions)]

        random_num = np.random.rand()


        
        if (random_num <= cdfValid[0]):
            print(directions[0], end=' ')
            state = moves[0]
        elif (random_num <= cdfValid[1]):
            print(directions[1], end=' ')
            state = moves[1]
        elif (random_num <= cdfValid[2]):
            print(directions[2], end=' ')
            state = moves[2]
        elif (random_num <= cdfValid[3]):
            print(directions[3], end=' ')
            state = moves[3]

decoder(sys.argv[1],sys.argv[2],sys.argv[3])