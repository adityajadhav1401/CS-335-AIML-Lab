import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
	def __init__(self, state, action, path_cost, parent_node, depth):
		self.state = state
		self.action = action
		self.path_cost = path_cost
		self.parent_node = parent_node
		self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
	"""
	Search the deepest nodes in the search tree first.
	"""

	def convertStateToHash(values):
		""" 
		values as a dictionary is not hashable and hence cannot be used directly in the explored set.
		This function changes values dict into a unique hashable string which can be used in the explored set.
		"""
		l = list(sorted(values.items()))
		modl = [a+b for (a, b) in l]
		return ''.join(modl)

	# YOUR CODE HERE

	frontier = util.Stack()
	explored = []

	rootNode = Node(problem.getStartState(), None, 0, None, 0)
	frontier.push(rootNode)

	while not frontier.isEmpty():
		currNode = frontier.pop()

		if(problem.isGoalState(currNode.state)):
			return currNode.state

		explored.append(currNode.state)

		for successor in problem.getSuccessors(currNode.state):
			if (successor[0] != False) and (successor[0] not in explored):
				nextState = successor[0]
				action = successor[1]
				path_cost =  successor[2] + currNode.path_cost 
				depth = 1 + currNode.depth
				successorNode = Node(nextState, action, path_cost, currNode, depth)
				frontier.push(successorNode)

	print "No solution found by DFS"
	return []

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.  This heuristic is trivial.
	"""
	return 0

def heuristic(state, problem):
	# It would take a while for Flat Earther's to get accustomed to this paradigm
	# but hang in there.
	curr_state_coord = tuple((tuple((problem.G.node[state]['x'],0,0)),tuple((problem.G.node[state]['y'],0,0)))) 
	end_state = problem.end_node
	end_state_coord = tuple((tuple((problem.G.node[end_state]['x'],0,0)),tuple((problem.G.node[end_state]['y'],0,0))))
	distance = util.points2distance(end_state_coord,curr_state_coord)

	return distance

def AStar_search(problem, heuristic=nullHeuristic):

	"""Search the node that has the lowest combined cost and heuristic first."""

	frontier = util.PriorityQueue()
	explored = []

	rootNode = Node(problem.getStartState(), None, 0, None, 0)
	frontier.push(rootNode, rootNode.path_cost + heuristic(rootNode.state, problem))

	while not frontier.isEmpty():
		currNode = frontier.pop()
		if currNode.state in explored:
			continue	

		if(problem.isGoalState(currNode.state)):
			path = [currNode]
			while True:
				if (path[-1].state == problem.getStartState()):break
				path.append(path[-1].parent_node)
			path.reverse()
			path = [node.state for node in path]
			return path

		explored.append(currNode.state)

		for successor in problem.getSuccessors(currNode.state):
			if successor[0] not in explored:
				nextState = successor[0]
				action = successor[1]
				path_cost =  successor[2] + currNode.path_cost 
				depth = 1 + currNode.depth
				successorNode = Node(nextState, action, path_cost, currNode, depth)
				frontier.update(successorNode, successorNode.path_cost + heuristic(successorNode.state, problem))
	print "No solution found by aStar"
	return []