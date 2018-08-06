#!/usr/bin/env python
from __future__ import division, print_function
import agents
import argparse
import reward
import datetime
import numpy as np
import scenarios
import sys
import time
import os.path
import plot_stats

try:
	import cPickle as pickle

except:
	import pickle

## Each failed test cases indicates a different failure in the system under test. This is not always true. One test case can fail due to multiple failures in the system and one failure can lead to 
## multiple failing test cases. This information is not easily available. Nevertheless, this approach tries to find all failing test cases and thereby indirectly also all detectable
## failures. 


## Future Proposals
## To address the threat, if possible, include failure causes as input features in future work.


## Write file is just to write the final outputs in a file
# write_file = open(" res.txt","a")
# comparison_file = open("compare.txt","w")

start_time = time.clock()

DEFAULT_NO_SCENARIOS = 1000
## No of Actions are defined for Tableau Representation for formation of table (Table formed is action_size * no_of_states)

DEFAULT_NO_ACTIONS = 100
## History Length is the length of the last verdicts that are considered for representing a state

DEFAULT_HISTORY_LENGTH = 4

DEFAULT_STATE_SIZE = DEFAULT_HISTORY_LENGTH + 1
## Learning Rate and Epsilon are Used for Tableau Form 

DEFAULT_LEARNING_RATE = 0.05

DEFAULT_EPSILON = 0.2
## Dump INterval is the interval or the build after which the stats are written to a file

DEFAULT_DUMP_INTERVAL = 1
DEFAULT_VALIDATION_INTERVAL = 0
## It signifies whether we want to print the stats or not

DEFAULT_PRINT_LOG = True
## It signifies whether we want to plot the graphs or not

DEFAULT_PLOT_GRAPHS = True
DEFAULT_NO_HIDDEN_NODES = 32
DEFAULT_TODAY = datetime.datetime.today()


log_file = open("retecs_log.txt","w")

def recency_weighted_avg(values, alpha):
	return sum(np.power(alpha, range(0, len(values))) * values) / len(values)


## Preprocess Function are Used for State Representation.They are used for representing a test case as a state . 
## Preprocess Continuous gives continuous values to the time Since and the duration group.
## State is represented as  [Duration Group , Time Since, Last histlen Verdicts]


## Input:This function takes the test case (The representaion of CSV file) , the history lengths and the Scenario Metadata (MaxExecTime , MaxDuration ) of a Cycle 
## Output : Returns the state representation which consists of [Duration Group , Time_Since Factor , Latest Verdicts (history length)]
def preprocess_continuous(state, scenario_metadata, histlen):

	## Time Slice is dependant on last run time, while time duration is dependant on the current duration of the test
	## Different Continuous Functions can be used here depending upon the data relation and the use case.
	if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
		time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
			scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
	else:
		time_since = 0

	history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

	if len(history) < histlen:
		history.extend([1] * (histlen - len(history)))

	if scenario_metadata['maxDuration'] > scenario_metadata['minDuration']:
			duration = (scenario_metadata['maxDuration'] - state['Duration']) / (scenario_metadata['maxDuration'] - scenario_metadata['minDuration'])
	else:
		duration = 0

	row = [
		# duration,
		state['Duration'] / scenario_metadata['totalTime'],
		time_since
	]
	row.extend(history)

	return tuple(row)

## Preprocess Function are Used for State Representation.They are used for representing a test case as a state . 
## Preprocess Discrete gives discrete values to the time Since and the duration group.
## State is represented as  [Duration Group , Time Since, Last histlen Verdicts]


## Input:This function takes the test case (The representaion of CSV file) , the history lengths and the Scenario Metadata (MaxExecTime , MaxDuration ) of a Cycle 
## Output : Returns the state representation which consists of [Duration Group , Time_Since Factor , Latest Verdicts (history length)]
def preprocess_discrete(state, scenario_metadata, histlen):
	## Time Slice is dependant on last run time, while time duration is dependant on the current duration of the test
	## The thresholds can be changed and the no of distinct levels can be increased.
	if scenario_metadata['maxDuration'] > scenario_metadata['minDuration']:
		duration = (scenario_metadata['maxDuration'] - state['Duration']) / (scenario_metadata['maxDuration'] - scenario_metadata['minDuration'])
	else:
		duration = 0

	if duration > 0.66:
		duration_group = 2
	elif duration > 0.33:
		duration_group = 1
	else:
		duration_group = 0

	## Tried to Discretize the duration group by having more levels, didn't improve the NAPFD and the Recall, though.
	# if duration > 0.8:
	# 	duration_group=2

	# elif duration >0.6 and duration <=0.8:
	# 	duration_group=1.5

	# elif duration >0.4 and duration<=0.6:
	# 	duration_group=1

	# elif duration >0.2 and duration <=0.4:
	# 	duration_group=0.5

	# else:
	# 	duration_group=0

	if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
		time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
			scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
	else:
		time_since = 0

	if time_since > 0.66:
		time_group = 2
	elif time_since > 0.33:
		time_group = 1
	else:
		time_group = 0

	history = [1 if res else 0 for res in state['LastResults'][0:histlen]]
	if len(history) < histlen:
		history.extend([1] * (histlen - len(history)))

	row = [
		duration_group,
		time_group
	]
	row.extend(history)

	# print(row)
	# input()
	return tuple(row)

## It takes the sc , the preprocess function and the agent as input.
## Sc contains all the relevant information about a particular build.
## Returns the results after the prioritization and selection of the test cases.
## Results include the no of detected failures, no of missed failures ,ttf, the napfd , the recall ,the precision(No significance here) and the detection ranks..
def process_scenario(agent, sc, preprocess):

	## Returns the Available agents, the total time, the minExec time, the max Exectime,the scheduleDatem, the min Duration and the max Duraion for the test cases of a particular build
	scenario_metadata = sc.get_ta_metadata()
	 
	# Scenario Metadat consists of various fieds having informations about the max Exec Duration,min Exec Duration etc .
	if agent.single_testcases:
		for row in sc.testcases():	
			# Build input vector: preprocess the observation
			# Preprocess function used here is preprocess_discrete and x is the state representation
			# The state representation is in the form of a tuple of Duration group, Time Slice and the last 4 execution results (if history length i 4)
			x = preprocess(row, scenario_metadata, agent.histlen) 

			## get_Action takes a state as input and assigns a priority to that state 
			action = agent.get_action(x)
			row['CalcPrio'] = action  # Store prioritization
	else:

		# Get all actions is used for heuristic sort and  Weighted average not for Network
		states = [preprocess(row, scenario_metadata, agent.histlen) for row in sc.testcases()]
		actions = agent.get_all_actions(states)
 
		# Setting the Priority of the different test cases (Index is the position of occuring of the test case in the test suite)
		for (tc_idx, action) in enumerate(actions):
			sc.set_testcase_prio(action, tc_idx)

	# Submit prioritized file for evaluation
	# step the environment and get new measurements
	## After the priorities have been assigned to the test cases, the selection of test casses is done depending upon the priorities and the available time.
	## The Results are returned .
	## Results include the no of detected failures, no of missed failures ,ttf, the napfd , the recall ,the precision(No significance here) and the detection ranks..

	return sc.submit()



class PrioLearning(object):
	def __init__(self, agent, scenario_provider, file_prefix, reward_function, output_dir, preprocess_function,
				 dump_interval=DEFAULT_DUMP_INTERVAL, validation_interval=DEFAULT_VALIDATION_INTERVAL):
		## Agent Refers to whether it is Network , Heur_Random , Heur_Sort ,Heur_weighted or Tableau
		self.agent = agent
		self.scenario_provider = scenario_provider
		
		## Reward Function is the reward function chosen for the RL algorithm
		self.reward_function = reward_function

		## Preprocess Function -- Already Explained
		self.preprocess_function = preprocess_function
		self.validation_res = []
		self.dump_interval = dump_interval
		self.validation_interval = validation_interval

		self.today = DEFAULT_TODAY

		self.file_prefix = file_prefix

		## Path of files to store the logs.
		self.val_file = os.path.join(output_dir, '%s_val' % file_prefix)
		self.stats_file = os.path.join(output_dir, '%s_stats' % file_prefix)
		self.agent_file = os.path.join(output_dir, '%s_agent' % file_prefix)


	#  For Output, refer to the below portion .
	## Not Applicabe for IndustrialDatasetScenarioProvider
	def run_validation(self, scenario_count):
		val_res = self.validation()

		for (key, res) in val_res.items():
			res = {
				'scenario': key,
				'step': scenario_count,
				'detected': res[0],
				'missed': res[1],
				'ttf': res[2],
				'napfd': res[3],
				'recall': res[4],
				'avg_precision': res[5]
				# res[4] are the detection ranks
			}

			self.validation_res.append(res)

	def validation(self):
		self.agent.train_mode = False
		val_scenarios = self.scenario_provider.get_validation()
		keys = [sc.name for sc in val_scenarios]
		results = [self.process_scenario(sc)[0] for sc in val_scenarios]
		self.agent.train_mode = True
		return dict(zip(keys, results))

	## It takes the scenario as input and returns the result of the prioritization and the reward given by the reward function.
	def process_scenario(self, sc):
		result = process_scenario(self.agent, sc, self.preprocess_function)
		reward = self.reward_function(result, sc)
		self.agent.reward(reward)
		return result, reward

	def replay_experience(self, batch_size):
		batch = self.replay_memory.get_batch(batch_size)

		for sc in batch:
			(result, reward) = self.process_scenario(sc)
			print('Replay Experience: %s / %.2f' % (result, np.mean(reward)))
			# log_file.write('Replay Experience: %s / %.2f' % (result, np.mean(reward)) )
			# log_file.write('\n')

	def train(self, no_scenarios, print_log, plot_graphs, save_graphs, collect_comparison=True):
		# Stats is the Dictionary of this object which has various useful fields mentioned below
		stats = {
			'scenarios': [],
			'rewards': [],
			'durations': [],
			'detected': [],
			'missed': [],
			'ttf': [], 
			'napfd': [],
			'recall': [],
			'avg_precision': [],
			'result': [],
			'step': [],
			'env': self.scenario_provider.name,
			'agent': self.agent.name,
			# 'action_size': self.agent.action_size,
			'history_length': self.agent.histlen,
			'rewardfun': self.reward_function.__name__,
			'sched_time': self.scenario_provider.avail_time_ratio,
			'hidden_size': 'x'.join(str(x) for x in self.agent.hidden_size) if hasattr(self.agent, 'hidden_size') else 0
		}

		if collect_comparison:
			cmp_agents = {
				'heur_sort': agents.HeuristicSortAgent(self.agent.histlen),
				'heur_weight': agents.HeuristicWeightAgent(self.agent.histlen),
				'heur_random': agents.RandomAgent(self.agent.histlen)
			}

			stats['comparison'] = {}

			# stats['comparison']['heur_sort/heur_weight/rand'] initialized for comparison 
			for key in cmp_agents.keys():
				stats['comparison'][key] = {
					'detected': [],
					'missed': [],
					'ttf': [],
					'napfd': [],
					'recall': [],
					'avg_precision': [],
					'durations': []
				}

		sum_actions = 0
		sum_scenarios = 0
		sum_detected = 0
		sum_missed = 0
		sum_reward = 0

		# Enumerate forms a tuple of (count,Element)
		# write_file.write("Agent is "+str(self.agent))

		for (i, sc) in enumerate(self.scenario_provider, start=1):
			if i > no_scenarios:
				break

			start = time.time()

			if print_log:
				print('ep %d:\tscenario %s\t' % (sum_scenarios + 1, sc.name), end='')
		

			(result, reward) = self.process_scenario(sc)

			end = time.time()

			# Statistics of the CI cycle after Prioritization and Selection of test cases from the test suite

			sum_detected += result[0]
			sum_missed += result[1]

			# In future if we want to include the priority of the test cases also, np.average() could be used, where weights can be adjusted depending 
			# upon the priority of the test cases

			sum_reward += np.mean(reward)
			sum_actions += 1
			sum_scenarios += 1
			duration = end - start

			stats['scenarios'].append(sc.name)
			stats['rewards'].append(np.mean(reward))
			stats['durations'].append(duration)
			stats['detected'].append(result[0])
			stats['missed'].append(result[1])
			 ## TTF is the time to failure or in simple words, it is the position at which the first test case that failed was placed by our algorithm
			stats['ttf'].append(result[2]) 
			stats['napfd'].append(result[3])
			stats['recall'].append(result[4])
			stats['avg_precision'].append(result[5])
			stats['result'].append(result)
			stats['step'].append(sum_scenarios)
	
			if print_log:
				print(' finished, reward: %.2f,\trunning mean: %.4f,\tduration: %.1f,\tresult: %s' %
					  (np.mean(reward), sum_reward / sum_scenarios, duration, result))
				
			global total_failures_detected
			global total_failures_missed
			total_failures_detected += result[0]
			total_failures_missed +=result[1]
			

			# Collect Comparison becomes True if we set args.comparable as True
			## Formulates the results of the heur_sort, heur_random and heur_weight . 

			if collect_comparison:
				for key in stats['comparison'].keys():
					start = time.time()
					cmp_res = process_scenario(cmp_agents[key], sc, preprocess_discrete)

					end = time.time()
					stats['comparison'][key]['detected'].append(cmp_res[0])
					stats['comparison'][key]['missed'].append(cmp_res[1])
					stats['comparison'][key]['ttf'].append(cmp_res[2])
					stats['comparison'][key]['napfd'].append(cmp_res[3])
					stats['comparison'][key]['recall'].append(cmp_res[4])
					stats['comparison'][key]['avg_precision'].append(cmp_res[5])
					stats['comparison'][key]['durations'].append(end - start)

			# # Data Dumping

			## The below two commented lines of code write the stats into the file after certain interval. No specific need of this, because anyways, we are writing the entire stats
			## at the end of the program

			# if self.dump_interval > 0 and sum_scenarios % self.dump_interval == 0:
			# 	pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

			if self.validation_interval > 0 and (sum_scenarios == 1 or sum_scenarios % self.validation_interval == 0):
				if print_log:
					print('ep %d:\tRun test... ' % sum_scenarios, end='')
		
				self.run_validation(sum_scenarios)

				pickle.dump(self.validation_res, open(self.val_file + '.p', 'wb'))

				if print_log:
					print('done')

		## Dumping the Stats of all the CI Cycles into the Stats_File
		if self.dump_interval > 0:
			self.agent.save(self.agent_file)
			pickle.dump(stats, open(self.stats_file + '.p', 'wb'))


		## Plotting Graphs 
		if plot_graphs:
			plot_stats.plot_stats_single_figure(self.file_prefix, self.stats_file + '.p', self.val_file + '.p', 1,
												plot_graphs=plot_graphs, save_graphs=save_graphs)
		## Save the generated Graphs
		if save_graphs:
			plot_stats.plot_stats_separate_figures(self.file_prefix, self.stats_file + '.p', self.val_file + '.p', 1,
												   plot_graphs=False, save_graphs=save_graphs)

		return np.mean(stats['napfd']),np.mean(stats['recall'])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	## Different Types of Prioritization Techniques
	parser.add_argument('-a', '--agent',
						choices=['tableau', 'network', 'heur_random', 'heur_sort', 'heur_weight'], default='network')
	## Dataset to be worked on
	## Siemens_Data_180 is the data for 180 CI cycles for a Project of Siemens
	parser.add_argument('-sp', '--scenario-provider',
						choices=['random', 'incremental', 'paintcontrol', 'iofrol', 'gsdtsr','siemens_data_30','siemens_data_180','hadoop_final_test','xml_test'], default='xml_test')
	
	## Selecting the appropriate reward function
	parser.add_argument('-r', '--reward', choices=['binary', 'failcount', 'timerank', 'tcfail'], default='timerank') 
	parser.add_argument('-p', '--prefix')

	## Selecting the length of the recent history verdicts that are considered for state representation
	parser.add_argument('-hist', '--histlen', type=int, default=DEFAULT_HISTORY_LENGTH)

	## Specifying the Hyperparameters which will be used in the Tavleau Representation (Not for the ANN Representation)
	parser.add_argument('-eps', '--epsilon', type=float, default=DEFAULT_EPSILON)
	parser.add_argument('-lr', '--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
	parser.add_argument('-as', '--actions', type=int, default=DEFAULT_NO_ACTIONS)

	## Specifying the no of Hidden Nodes for the ANN
	parser.add_argument('-ns', '--hiddennet', type=int, default=DEFAULT_NO_HIDDEN_NODES)
	parser.add_argument('-n', '--no-scenarios', type=int, default=DEFAULT_NO_SCENARIOS)
	parser.add_argument('-d', '--dump_interval', type=int, default=DEFAULT_DUMP_INTERVAL)
	parser.add_argument('-v', '--validation_interval', type=int, default=DEFAULT_VALIDATION_INTERVAL)
	parser.add_argument('-o', '--output_dir', default='.')
	parser.add_argument('-q', '--quiet', action='store_true', default=False)
	
	## Generation and Saving of Graphs is done by specifying the default as True here.
	parser.add_argument('--plot-graphs', action='store_true', default=True)
	parser.add_argument('--save-graphs', action='store_true', default=False)

	##Comapes the chosen memory representation and the reward function with the three algorithms (Heur_Sort, Heur_Weight, Heur_Random)
	parser.add_argument('--comparable', action='store_true', default=True)
	args = parser.parse_args()

	## State is represented by [Duration Gro-up , Time Slice , Last Histlen Verdicts] 
	## Hence the state size is 2 + hislen 
	## In future Other Important features may also be added to represent the state e.g the priority etc. and the state size has to be changed then.
	state_size = 2 + args.histlen

	## Preprocess Function are Used for State Representation.They are used for representing a test case as a state . 
	## Preprocess Continuous gives continuous values to the time group and the duration slice.
	## Preprocess Discrete gives discrete values [0,1,2] to the time group and the duration state.
	preprocess_function = preprocess_discrete

	## Initializing the agent
	if args.agent == 'tableau':
		agent = agents.TableauAgent(learning_rate=args.learning_rate, state_size=state_size, action_size=args.actions,
									epsilon=args.epsilon, histlen=args.histlen)
	

	## If the action size is 1 , we use MLPClassifier 
	## for action size as 2, we use MLPRegressor
	elif args.agent == 'network':
		if args.reward in ('binary'):
			action_size = 1
		else:
			action_size = 2

		agent = agents.NetworkAgent(state_size=state_size, action_size=action_size, hidden_size=args.hiddennet,
									histlen=args.histlen)
	elif args.agent == 'heur_random':
		agent = agents.RandomAgent(histlen=args.histlen)
	elif args.agent == 'heur_sort':
		agent = agents.HeuristicSortAgent(histlen=args.histlen)
	elif args.agent == 'heur_weight':
		agent = agents.HeuristicWeightAgent(histlen=args.histlen)
	else:
		print('Unknown Agent')
		sys.exit()

	if args.scenario_provider == 'random':
		scenario_provider = scenarios.RandomScenarioProvider()
	elif args.scenario_provider == 'incremental':
		scenario_provider = scenarios.IncrementalScenarioProvider(episode_length=args.no_scenarios)
	elif args.scenario_provider == 'paintcontrol':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/paintcontrol.csv')
		# scenario_provider = scenarios.FileBasedSubsetScenarioProvider(scheduleperiod=datetime.timedelta(days=1),
																	  # tcfile='paintcontrol.csv',
																	  # solfile='paintcontrol.csv')
		args.validation_interval = 0
	elif args.scenario_provider == 'iofrol':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/iofrol.csv')
		args.validation_interval = 0
	elif args.scenario_provider == 'gsdtsr':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/gsdtsr.csv')
		args.validation_interval = 0

	elif args.scenario_provider == 'siemens_data':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/siemens_data.csv')
		args.validation_interval = 0

	elif args.scenario_provider == 'siemens_data_30':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/siemens_data_30.csv')
		args.validation_interval = 0

	elif args.scenario_provider == 'siemens_data_90':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/siemens_data_90.csv')
		args.validation_interval = 0
	elif args.scenario_provider == 'siemens_data_180':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/siemens_data_180.csv')
		args.validation_interval = 0

	elif args.scenario_provider == 'hadoop_final_test':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/hadoop_final_test.csv')
		args.validation_interval = 0

	elif args.scenario_provider == 'xml_test':
		scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/xml_test.csv')
		args.validation_interval = 0
	
	if args.reward == 'binary':
		reward_function = reward.binary_positive_detection_reward
	elif args.reward == 'failcount':
		reward_function = reward.failcount
	elif args.reward == 'timerank':
		reward_function = reward.timerank
	elif args.reward == 'tcfail':
		reward_function = reward.tcfail


	# write_file.write("Reward Function is " + str(args.reward))

	prefix = '{}_{}_{}_lr{}_as{}_n{}_eps{}_hist{}_{}'.format(args.agent, args.scenario_provider, args.reward,
															 args.learning_rate, args.actions, args.no_scenarios,
															 args.epsilon, args.histlen, args.prefix)

	total_failures_present=0
	total_failures_detected=0
	total_failures_missed=0

	rl_learning = PrioLearning(agent=agent,
							   scenario_provider=scenario_provider,
							   reward_function=reward_function,
							   preprocess_function=preprocess_function,
							   file_prefix=prefix,
							   dump_interval=args.dump_interval,
							   validation_interval=args.validation_interval,
							   output_dir=args.output_dir)

	## Avg_napfd is the average NAPFD after prioritization and selection of test cases by the algorithm selected across all the CI cycles of the dataset
	## rec is the mean Recall (Not the actual recall)
	## rec can be ignored, it was just used for observing the behaviour
	avg_napfd,rec = rl_learning.train(no_scenarios=args.no_scenarios, print_log=not args.quiet,
								  plot_graphs=args.plot_graphs,
								  save_graphs=args.save_graphs,
								  collect_comparison=args.comparable)
	
	## Two different types of evaluation metrics were used . The NAPFD average and the RECALL (% of failures detected). The appropriate evauation metric that needs to be mximized depends upon the
	## use case. Similarly, Other factors can also be incorporated while evaluating the algorithms if the priority of the test cases in known.
	total_failures_present = total_failures_detected + total_failures_missed
	print(total_failures_present)
	print(total_failures_detected)
	print(total_failures_missed)
	print("Recall is ",(100.0*total_failures_detected)/total_failures_present)
	print(avg_napfd)
	end_time=time.clock()
	print("The total time taken is ", end_time - start_time)