# Reinforced Test Case Selection (RETECS)

Implementation of RETECS and data sets for the paper *Reinforcement Learning for Automatic Test Case Prioritization and Selection in Continuous Integration*.

## Abstract

> Testing in Continuous Integration (CI) involves test case prioritization, selection, and execution at each cycle.
> Selecting the most promising test cases to detect bugs is hard if there are uncertainties on the impact of committed code changes and traceability links between code and tests are not available.
> This paper introduces RETECS, a new method for automatically learning test case selection and prioritization in CI with the goal to minimize round-trip-times between code commits and developer feedback on failed test cases.
> The RETECS method uses reinforcement learning to select and prioritize test cases according to their duration, previous last execution and failure history.
> In a constantly changing environment, where new test cases are created and obsolete test cases are deleted, the RETECS method learns to prioritize error-prone test cases higher under guidance of a reward function and by observing previous CI cycles.
> By applying RETECS on data extracted from three industrial case studies, we show for the first time that reinforcement learning enables fruitful automatic adaptive test case selection and prioritization in CI and regression testing. 

# Configuration

## RETECS

`retecs.py` is the main file of the project.
It provides command line arguments to control the execution.
Run `./retecs.py -h` for an overview of all arguments and a description.

## Experiments

Common configuration is located in `run_experiment_common.py`, but can be overwritten from each experiment file.

- `USE_LATEX`: If True, figures are plotted by xelatex (has to be installed separately), else the standard matplotlib backend is used (default: False)
- `DATA_DIR`: Directory to store raw results of experiments (default: RESULTS)
- `FIGURE_DIR`: Directory to store exported figures from evaluation (default: RESULTS)
- `ITERATIONS`: Number of times the experiment is repeated (default: 30)
- `PARALLEL`: Run iterations in parallel (default: True)
- `PARALLEL_POOL_SIZE`: Number of parallel executions (default: 6)

# How to run

## Using your local Python installation

1. (Optionally, but recommended) Create a virtual environment and activate it
`$ virtualenv2 venv_retecs` and `$ source venv_retecs/bin/activate`

2. Install requirements `$ pip install -r requirements.txt`

3. Run an experiment: a) All experiments: `$ ./run_all_experiments.sh` or b) a specific experiment: Call experiment file, e.g. `$ ./run_experiment_rq1.py`

## Using Docker

Build the image: `$ docker build -t retecs .`

Run all experiments: `$ docker run -it --rm -v "$PWD":/usr/src/app -w /usr/src/app --name run_all_experiments retecs ./run_all_experiments.sh`

Run a specific experiment (replace $EXPERIMENT with the experiment file): `$ docker run -it --rm -v "$PWD":/usr/src/app -w /usr/src/app --name $EXPERIMENT retecs ./$EXPERIMENT`, e.g. `$ docker run -it --rm --name run_experiment_rq1.py -v "$PWD":/usr/src/app -w /usr/src/app retecs ./run_experiment_rq1.py`

Results are placed in the local RESULTS directory.


## RETECS

retecs.py is the main file. Change the Memory Representaion and the reward function in this file and run it to see the results. To plot the graphs, assign plot_graph variable to True and to compare the algorithm with other deterministic algorithms, assign comparable variable as True.

# Structure

| File/Directory | Content |
| -------- | ------- |
| csvconvert | Code to convert the JSON dump file of Siemens Data into the required format after using hashmaps and formulating the Results and the Last Run History.
| jenkins_extract_data.py | Code to Extract the test case build data from Jenkings , Convert it into the required format and then write the formatted data into a CSV File.
|Report| It contains the detailed report of RETECS
| DATA | Industrial data sets: Paint Control, IOF/ROL, GSDTSR , Siemens Data and Hadoop Data |
| RESULTS | Results of experimental runs are placed here, both raw data and visualization |
| requirements.txt | Python requirements of RETECS |
| retecs.py | Main file for training and running the RETECS algorithm |
| reward.py | Reward functions |
| run_all_experiments.sh | Run all experiments (calls each experiment script below) |
| run_experiment_common.py | Common configuration and functions for experiment runners |
| run_experiment_rq0_history_length.py | Evaluate history length influence |
| run_experiment_rq0_network_hidden_nodes.py | Evaluate sizes of neural network |
| run_experiment_rq0_reward_function.py | Compare reward functions on same data set |
| run_experiment_rq0_scheduling_time.py | Evaluate different times for test execution per CI cycle |
| run_experiment_rq0_tableau_action_size.py | Evaluate different numbers of actions for the Tableau-based RL agent |
| run_experiment_rq1.py | Run all combinations of RL agents and reward functions on three data sets, evaluate their learning performance |
| run_experiment_rq2.py | Run all combinations of RL agents and reward functions on three data sets, evaluate their performance compared to basic prioritization functions |
| scenarios.py | Main Class 'IndustrialDataSetProvider': Reads data sets, provides CI cycles, evaluates created schedules |
| stats.py | Helper functions for evaluation |
