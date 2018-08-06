import numpy as np
import os
from sklearn import neural_network

try:
	import cPickle as pickle
except:
	import pickle



## Experienec replay is used for remembering the experience .Experience consists of (state,reward) tuples.
## It also has a function which returns a batch of random experiences of batch_size out of its saved experience.    
class ExperienceReplay(object):
	def __init__(self, max_memory=5000, discount=0.9):
		self.memory = []
		## Max_memory is the maximum no of experiences it can store
		self.max_memory = max_memory
		self.discount = discount

	## It stores the experiences into the memory
	def remember(self, experience):
		## Experience consists of (state,reward) tuples.
		self.memory.append(experience)

	## A batch containing batch_size no of ramdom experiences is returned by the function.
	## Experience is a tuple of (state,reward)
	def get_batch(self, batch_size=10):
		if len(self.memory) > self.max_memory:
			del self.memory[:len(self.memory) - self.max_memory]

		if batch_size < len(self.memory):
			timerank = range(1, len(self.memory) + 1)
			p = timerank / np.sum(timerank, dtype=float)
			## p is the probability of selecting a particular experience
			batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
			batch = [self.memory[idx] for idx in batch_idx]
		else:
			batch = self.memory
		return batch


class BaseAgent(object):
	def __init__(self, histlen):
		self.single_testcases = True
		self.train_mode = True
		self.histlen = histlen

	def get_action(self, s):
		return 0

	def get_all_actions(self, states):
		""" Returns list of actions for all states """
		return [self.get_action(s) for s in states]

	def reward(self, reward):
		pass

	def save(self, filename):
		""" Stores agent as pickled file """
		pickle.dump(self, open(filename + '.p', 'wb'), 2)

	@classmethod
	def load(cls, filename):
		return pickle.load(open(filename + '.p', 'rb'))

##Tavleau Based Memory Representations
## For each state, two lists of each of action_size length are formed. One stores the Q-value of the state-action pair
## The other table stores the count of the no of times a particlar action is taken at that particular state . It is used while updating the Q-values.
class TableauAgent(BaseAgent):
	def __init__(self, learning_rate, state_size, action_size, epsilon, histlen):
		# Key: (State Representation) -> (N, Q)
		super(TableauAgent, self).__init__(histlen=histlen)
		self.name = 'tableau'
		self.state_in = state_size
		self.states = {}  # The Tableau
		## Initialization of q values
		self.initial_q = 5
		self.action_history = []
		## Action size 
		self.action_size = action_size
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.max_epsilon = epsilon
		## The minimum epsilon
		self.min_epsilon = 0.1
		self.gamma = 0.99

	## Takes a state as input and returns a priority(action) to it.
	def get_action(self, s):
		if s not in self.states:
			self.states[s] = {
				'Q': [self.initial_q] * self.action_size,
				'N': [0] * self.action_size
			}

		## If rand() gives a number greater than epsilon ,we choose the best action (The one with the maximum reward)
		if np.random.rand() >= self.epsilon:
			action = self.random_argmax(self.states[s]['Q'])
		
		## Otherwise, we choose a random action	
		else:
			action = np.random.randint(self.action_size)

		## Appending the action history
		if self.train_mode:
			self.action_history.append((s, action))

		return action

	## Q values and N values for the State-Action pair are updated based on the rewards given by the scenario ..
	def reward(self, rewards):
		if not self.train_mode:
			return

		## Reward for the different state and the actions performed (given by the environment)
		try:
			x = float(rewards)
			rewards = [x] * len(self.action_history)
		except:
			if len(rewards) < len(self.action_history):
				raise Exception('Too few rewards')

		# Update Q values and the count values After getting the rewards
		for ((state, act_idx), reward) in zip(self.action_history, rewards):
			self.states[state]['N'][act_idx] += 1
			n = self.states[state]['N'][act_idx]
			prev_q = self.states[state]['Q'][act_idx]
			self.states[state]['Q'][act_idx] = prev_q + 1.0 / n * (reward - prev_q)
			# self.states[state]['Q'][act_idx] = prev_q + self.learning_rate * (reward - prev_q)

		self.reset_action_history()
		self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon

	## The action history is cleared
	def reset_action_history(self):
		self.action_history = []

	@staticmethod
	def random_argmax(vector):
		""" Argmax that chooses randomly among eligible maximum indices. """
		m = np.amax(vector)
		indices = np.nonzero(vector == m)[0]
		return np.random.choice(indices)

## Neural Nets are used which are adjusted depending upon the experiences (state, reward) tuples .
## The Neural Nets take a state (Duration Group,Time Since,Last History Length Verdicts) as input and returns its priority as output.
## A simple MLPRegressor is used here. More Complex ANN's can be used if the data is sufficient to improve the efficiency.
class NetworkAgent(BaseAgent):
	def __init__(self, state_size, action_size, hidden_size, histlen):
		super(NetworkAgent, self).__init__(histlen=histlen)
		self.name = 'mlpclassifier'

		## Experience Length is the maximum recent Experience that it remembers. If the experience length increases, the most past experiences are deleted.
		self.experience_length = 10000
		
		## Exp batch Size is the batch size that is used for training the ANN
		self.experience_batch_size = 1000
		
		self.experience = ExperienceReplay(max_memory=self.experience_length)
		self.episode_history = []
		self.iteration_counter = 0

		## Action Size is passed as the Parameter.
		self.action_size = action_size

		if isinstance(hidden_size, tuple):
			self.hidden_size = hidden_size
		else:
			self.hidden_size = (hidden_size,)
		self.model = None
		self.model_fit = False
		self.init_model(True)

	# TODO This could improve performance (if necessary)
	# def get_all_actions(self, states):
	#   try:

	def init_model(self, warm_start=True):
		## warm_start : When set to True, reuse the solution of the previous call to fit as initialization, 
		## otherwise, just erase the previous solution.

		if self.action_size == 1:
			self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
													  warm_start=warm_start, solver='adam', max_iter=1200)
		else:
			self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_size, activation='relu',
													 warm_start=warm_start, solver='adam', max_iter=1200)
		self.model_fit = False

	## Takes a state and returns the priority .
	def get_action(self, s):
		# Action is assigning a Priority to a state ( A test case alongwith the history and duration  and time group)
		if self.model_fit:
			if self.action_size == 1:
				a = self.model.predict_proba(np.array(s).reshape(1, -1))[0][1]
			else:
				a = self.model.predict(np.array(s).reshape(1, -1))[0]
		else:
			a = np.random.random()
			# a is a random priority assigned to it between 0 and 1

		## Episode History is a tuple of State and Action
		if self.train_mode:
			self.episode_history.append((s, a))

		return a

	def reward(self, rewards):
		if not self.train_mode:
			return

		try:
			x = float(rewards)
			rewards = [x] * len(self.episode_history)
		except:
			if len(rewards) < len(self.episode_history):
				raise Exception('Too few rewards')

		self.iteration_counter += 1

		## Remembering the experiences for trainnig the ANN
		for ((state, action), reward) in zip(self.episode_history, rewards):
			self.experience.remember((state, reward))

		self.episode_history = []

		## It adjusts the weights of the Neural Net after 5 iterations. 
		##Can be altered according to the data to improve the efficeiency of the algorithm  
		if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
			self.learn_from_experience()

	## Gets a Random batch of Experiences and the the weights of the model are adjusted accordingly.
	def learn_from_experience(self):
		experiences = self.experience.get_batch(self.experience_batch_size)
		x, y = zip(*experiences)

		if self.model_fit:
			try:
				self.model.partial_fit(x, y)
			except ValueError:
				self.init_model(warm_start=False)
				self.model.fit(x, y)
				self.model_fit = True
		else:
			# print("here")
			self.model.fit(x, y)  # Call fit once to learn classes
			self.model_fit = True



## Random Selection  
class RandomAgent(BaseAgent):
	def __init__(self, histlen):
		super(RandomAgent, self).__init__(histlen=histlen)
		self.name = 'random'

	def get_action(self, s):
		return np.random.random()

	def get_all_actions(self, states):
		prio = range(len(states))
		np.random.shuffle(prio)
		return prio


""" Sort first by last execution results, then time not executed """
class HeuristicSortAgent(BaseAgent):	

	def __init__(self, histlen):
		super(HeuristicSortAgent, self).__init__(histlen=histlen)
		self.name = 'heuristic_sort'
		self.single_testcases = False

	## Not Implemented 
	def get_action(self, s):
		raise NotImplementedError('Single get_action not implemented for HeuristicSortAgent')


	## states[x][-self.histlen-1] refers to the time_since 
	## Returns the Priorities of all the test cases 
	def get_all_actions(self, states):
		

		sorted_idx = sorted(range(len(states)),
							key=lambda x: list(states[x][-self.histlen:]) + [states[x][-self.histlen - 1]])

		## Sorted Actions 
		sorted_actions = sorted(range(len(states)), key=lambda i: sorted_idx[i])
		return sorted_actions


## It sorts according to weighted sum of the features of the state.

class HeuristicWeightAgent(BaseAgent):
	""" Sort by weighted representation """

	def __init__(self, histlen):
		super(HeuristicWeightAgent, self).__init__(histlen=histlen)
		self.name = 'heuristic_weight'
		self.single_testcases = False
		self.weights = []

	## Not Implemented 
	def get_action(self, s):
		raise NotImplementedError('Single get_action not implemented for HeuristicWeightAgent')

	## Returns the priorities of all the test cases
	def get_all_actions(self, states):
		if len(self.weights) == 0:
			state_size = len(states[0])

			## The weights of the heuristic weight can be adjusted depending upon the relation and the importance of the different parameters of the state to give better performance. 
			## e.g the last verdict should be givn more weightage as compared to the second-last verdict .
			self.weights = np.ones(state_size) / state_size
			# print(self.weights)
			# input()

		sorted_idx = sorted(range(len(states)), key=lambda x: sum(states[x] * self.weights))
		sorted_actions = sorted(range(len(states)), key=lambda i: sorted_idx[i])
		return sorted_actions

def restore_agent(model_file):
	if os.path.exists(model_file + '.p'):
		return BaseAgent.load(model_file)
	else:
		raise Exception('Not a valid agent')
