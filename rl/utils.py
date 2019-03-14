import os
import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import numpy as np
import cv2
import pandas as pd
import time
from tqdm import tqdm_notebook as tqdm
from collections import deque
import random
from collections import defaultdict


def one_hot(arr,C):
	''' NUMPY arr w/ shape [n] -> [n,C], where n - # of elements '''
	return np.eye(C)[[arr]] if isinstance(arr, int) else np.eye(C)[arr] # add extra dim if arr==number


def np_map(fn, list_):
	''' apply function to each element in list_ '''
	return np.array(list(map(fn, list_)), dtype=np.float32)


def discount_rewards(R, y=0.99, normalize=True):
	''' [n] -> [n] w/ discounted rewards '''
	G = np.zeros_like(R)
	for t in range(len(R)-1,-1,-1):
		G[t] = R[t] + y*G[t+1] if t+1 < len(R) else R[t]
	return (G - G.mean() ) / G.std() if normalize else G


'''
CLASSES
'''

class RandomAgent():
	def __init__(self, num_actions):
		self.A = num_actions

	def first_step(self, s): # current state
		return np.random.choice(self.A)

	def step(self, r, s): # prev state reward and current state
		return np.random.choice(self.A)

	def last_step(self, r): # prev state reward
		return np.random.choice(self.A)


class Experiment():
	def __init__(self, env, agent, use_tqdm=True):
		self.env = env
		self.agent = agent
		self.train_steps = 250000
		self.eval_steps = 125000
		self.max_steps_in_episode = 27000
		self.iter = 0
		self.tqdm = use_tqdm
		self.D = defaultdict(list) # dict for each iter
	
	
	def run(self, iters, steps_in_one_iter=250000, name='experiment', run_number=1.0): # used like in atari baselines
		''' 1 iter = run as many episodes till total_steps == steps_in_one_iter '''
		T = tqdm(range(iters)) if self.tqdm else range(iters)
		for _ in T:
			d = self._run_one_phase(max_steps=steps_in_one_iter) # returns self.E averaged across episodes
			self.D['run_number'].append(float(run_number))
			self.D['iteration'].append(float(self.iter))
			self.D['train_episode_returns'].append(float(d['reward/e'])) # reward per episode
			self.D['agent'].append(name)
			if self.tqdm:
				T.set_description('{:>4d}/{:>4d}) r={:>5.1f} steps={:5.1f}'.format(self.iter, iters, d['reward/e'], d['total_steps/e']))
			for k,v in d.items():
				self.D[k].append(v)
			self.iter += 1
		return pd.DataFrame(data=self.D)
		
	
	def _run_one_phase(self, max_steps):
		num_episodes = 0
		if self.tqdm and isinstance(self.env, AtariPreprocessing) and max_steps > 100 or max_steps > 20000:
			T = tqdm(total=max_steps)
		self.E = defaultdict(int) # save all printable vars in here (default value = 0)
		while self.E['total_steps'] < max_steps:
			steps = self._run_one_episode(self.max_steps_in_episode) # total reward here
			num_episodes += 1
			if self.tqdm and isinstance(self.env, AtariPreprocessing) and max_steps > 100 or max_steps > 20000:
				T.update(steps)
				s = 'steps) {:>5.1f};'.format(steps)
				D = []; [D.extend([a,b/self.E['total_steps']]) for a,b in self.E.items() if a != 'total_steps']
				T.set_description(s + ''.join([' {}) {:>5.1f};' for _ in len(self.E)]).format(D))
		d = {}
		for a,b in self.E.items():
			d[a+'/e'] = b/num_episodes
			d[a+'/s'] = b/num_episodes/self.E['total_steps']
		del self.E; return d
	

	def agent_state_preprocessing_fn(self, s):
		self.state_memory.append(np.squeeze(s))
		if len(self.state_memory) == self.state_memory.maxlen:
			s = np.array(self.state_memory) # [4,84,84]
			s = np.transpose(s, [1,2,0]) # [84,84,4]
			s = s.astype(np.float32)
			s = np.expand_dims(s, 0) # [1,84,84,4] add batch dim
			return s / 255.


	def _run_one_episode(self, max_steps, mode='train', delta_time=1/24): # mode = 'train' or 'eval' or 'watch'
		total_steps = 0
		# init env
		if isinstance(self.env, AtariPreprocessing): # concat frames
			num_frames = 4
			self.state_memory = deque([], maxlen=num_frames)
			init_obs = self.env.reset()
			self.agent_state_preprocessing_fn(init_obs)
			for _ in range(self.state_memory.maxlen-1):
				obs, r, done, _ = self.env.step(self.env.action_space.sample())
				if done: raise(Exception('WTFF'))
				obs = self.agent_state_preprocessing_fn(obs)
				self.E['total_steps'] += 1
			a = self.agent.first_step(obs, self.E)
		else:
			if isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
				self.agent_state_preprocessing_fn = lambda s: one_hot([s],C=self.env.observation_space.n)
			else:
				self.agent_state_preprocessing_fn = lambda s: np.expand_dims(s, 0)
			init_obs = self.env.reset()
			init_obs = self.agent_state_preprocessing_fn(init_obs) # add batch shape
			a = self.agent.first_step(init_obs, self.E)
			self.E['total_steps'] += 1
		done = False
		while True:
			if mode == 'watch':
				self.env.render('human')
				time.sleep(delta_time)
			a = a if isinstance(a, int) else int(np.squeeze(a)) # if a is list of one elem then it is valid
			obs, r, done, _ = self.env.step(a)
			obs = self.agent_state_preprocessing_fn(obs)
			r = np.clip(r, -1, 1)
			self.E['reward'] += r
			self.E['total_steps'] += 1
			total_steps += 1
			if total_steps > max_steps or done: break
			else: a = self.agent.step(r, obs, self.E)
		self.agent.last_step(r, self.E)
		self.E['total_steps'] += 1
		if mode == 'watch':
			if isinstance(self.env, AtariPreprocessing):
				self.env.environment.close()
			else:
				self.env.close()
		return total_steps


	def show_play(self, steps=100, delta_time=1/24): # 24 fps
		return self._run_one_episode(steps, 'watch', delta_time)
		
	



'''
UTIL FUNCTIONS
'''


ALL_GAMES = ['AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
			 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk',
			 'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede',
			 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
			 'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
			 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
			 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
			 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix',
			 'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid',
			 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris',
			 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham',
			 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge',
			 'Zaxxon'] # dopamine atari games

SIMPLE_GAMES = { # simple game:max iteration
	'CartPole-v1':500, 	# x1=1min x5=7min
#     'MountainCar-v0':0,
#     'Acrobot-v2':0,
	'Taxi-v2':500, 		# x1=2min x5=12min
	'NChain-v0':200 	# x1=2min x5=10min
}


def test_agent_on_simple_games(create_agent_fn, name, game_dict=SIMPLE_GAMES):
	for game, max_iters in tqdm(game_dict.items(), total=len(game_dict)):
		df = pd.DataFrame()
		for run in tqdm(range(5), leave=True):
			env = gym.make(game)
			if isinstance(env.observation_space, gym.spaces.box.Box):
				S, A = env.observation_space.shape[0], env.action_space.n
			else:
				S, A = env.observation_space.n, env.action_space.n
			exp = Experiment(env, create_agent_fn(S,A), use_tqdm=False)
			df2 = exp.run(iters=max_iters, steps_in_one_iter=1, name=name, run_number=run)
			df = df.append(df2)
		save_my_benchmark(df, game, name)
	return True


def save_my_benchmark(df, game_name, alg_name, base_dir='./benchmarks'):
	path = os.path.join(base_dir, 'my_benchmarks', game_name)
	os.makedirs(path, exist_ok=True)
	path = os.path.join(path, '{}.pkl'.format(alg_name))
	df.to_pickle(path)
	return True
	
def load_my_benchmark(game_name, agent_name=None, base_dir='./benchmarks'):
	if agent_name is None:
		df = pd.DataFrame()
		directory = os.path.join(base_dir, 'my_benchmarks', game_name)
		for filename in os.listdir(directory):
			if filename.endswith(".pkl") and not filename.startswith("agent"):
				df2 = pd.read_pickle(os.path.join(directory,filename))
				df = df.append(df2)
		return df
	else:
		path = os.path.join(base_dir, 'my_benchmarks', game_name, '{}.pkl'.format(agent_name))
		return pd.read_pickle(path)

def load_atari_benchmark(game_name):
	return load_baselines()[game_name]

def plot_df(df, y='train_episode_returns', confidence=68, yname=''):
	# yname is used as y axis name when y is list
	if isinstance(y, list):
		out = pd.DataFrame()
		for i,col in enumerate(y):
			d = df[ [col, 'iteration'] ]
			d.rename(columns={col:yname}, inplace=True)
			d.insert(0, 'agent', col) # add additional col w/ name
			out = out.append(d)
		return sns.lineplot(x='iteration', y=yname, hue='agent', ci=confidence, data=out)
	else:
		return sns.lineplot(x='iteration', y=y, hue='agent', ci=confidence, data=df)


# # DEPRICATED
# def create_new_reward_df(rewards, name='experiment'): # rewards.shape = (num_runs,num_iters) [num_runs can be just 1]
# 	d = {'run_number':[], 'iteration': [], 'train_episode_returns': [], 'agent': []}
# 	for run_nr, rewards_arr in enumerate(rewards):
# 		for i, r in enumerate(rewards_arr):
# 			d['iteration'].append(i)
# 			d['run_number'].append(run_nr+1)
# 			d['train_episode_returns'].append(r)
# 			d['agent'].append(name)
# 	return pd.DataFrame(data=d)


# # DEPRICATED
# def plot_results(rewards, game_name='Pong', name='experiment', baselines=True):
# 	# rewards.shape = (num_runs,num_iters) [num_runs can be just 1]
# 	df = create_new_reward_df(rewards, name)
# 	if baselines:
# 		b = load_baselines()
# 		df_b = b[game_name]
# 		df = df_b.append(df)
# 	fig, ax = plt.subplots(figsize=(16,8))
# 	sns.set_style("dark")
# 	sns.tsplot( data=df, time='iteration', unit='run_number', condition='agent', value='train_episode_returns', ax=ax )
# 	plt.title(game_name)
# 	return plt.show()


def load_baselines(base_dir='./benchmarks', verbose=False):
	"""Reads in the baseline experimental data from a specified base directory.

	Args:
	base_dir: string, base directory where to read data from.
	verbose: bool, whether to print warning messages.

	Returns:
	A dict containing pandas DataFrames for all available agents and games.
	"""
	experimental_data = {}
	for game in ALL_GAMES:
		for agent in ['dqn', 'c51', 'rainbow', 'iqn']:
			game_data_file = os.path.join(base_dir, agent, '{}.pkl'.format(game))
			with open(game_data_file, 'rb') as f:
				if sys.version_info.major >= 3:
					# pylint: disable=unexpected-keyword-arg
					single_agent_data = pickle.load(f, encoding='latin1')
					# pylint: enable=unexpected-keyword-arg
				else:
					single_agent_data = pickle.load(f)
				single_agent_data['agent'] = agent
				if game in experimental_data:
					experimental_data[game] = experimental_data[game].merge(
						single_agent_data, how='outer')
				else:
					experimental_data[game] = single_agent_data
	return experimental_data





'''
DOPAMINE FUNCTIONS AND CLASSES
'''

def create_atari_env(game_name=None, sticky_actions=True):
	"""Wraps an Atari 2600 Gym environment with some basic preprocessing.

	This preprocessing matches the guidelines proposed in Machado et al. (2017),
	"Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
	Problems for General Agents".

	The created environment is the Gym wrapper around the Arcade Learning
	Environment.

	The main choice available to the user is whether to use sticky actions or not.
	Sticky actions, as prescribed by Machado et al., cause actions to persist
	with some probability (0.25) when a new command is sent to the ALE. This
	can be viewed as introducing a mild form of stochasticity in the environment.
	We use them by default.

	Args:
		game_name: str, the name of the Atari 2600 domain.
		sticky_actions: bool, whether to use sticky_actions as per Machado et al.

	Returns:
		An Atari 2600 environment with some standard preprocessing.
	"""
	assert game_name is not None
	game_version = 'v0' if sticky_actions else 'v4'
	full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
	env = gym.make(full_game_name)
	# Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
	# handle this time limit internally instead, which lets us cap at 108k frames
	# (30 minutes). The TimeLimit wrapper also plays poorly with saving and
	# restoring states.
	env = env.env
	env = AtariPreprocessing(env)
	return env



class AtariPreprocessing(object):
	"""A class implementing image preprocessing for Atari 2600 agents.

	Specifically, this provides the following subset from the JAIR paper
	(Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

		* Frame skipping (defaults to 4).
		* Terminal signal when a life is lost (off by default).
		* Grayscale and max-pooling of the last two frames.
		* Downsample the screen to a square image (defaults to 84x84).

	More generally, this class follows the preprocessing guidelines set down in
	Machado et al. (2018), "Revisiting the Arcade Learning Environment:
	Evaluation Protocols and Open Problems for General Agents".
	"""

	def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
			   screen_size=84):
		"""Constructor for an Atari 2600 preprocessor.

		Args:
		environment: Gym environment whose observations are preprocessed.
		frame_skip: int, the frequency at which the agent experiences the game.
		terminal_on_life_loss: bool, If True, the step() method returns
			is_terminal=True whenever a life is lost. See Mnih et al. 2015.
		screen_size: int, size of a resized Atari 2600 frame.

		Raises:
		ValueError: if frame_skip or screen_size are not strictly positive.
		"""
		if frame_skip <= 0:
			raise ValueError('Frame skip should be strictly positive, got {}'.
						format(frame_skip))
		if screen_size <= 0:
			raise ValueError('Target screen size should be strictly positive, got {}'.
						format(screen_size))

		self.environment = environment
		self.terminal_on_life_loss = terminal_on_life_loss
		self.frame_skip = frame_skip
		self.screen_size = screen_size
		self.state_deque = deque()

		obs_dims = self.environment.observation_space
		# Stores temporary observations used for pooling over two successive
		# frames.
		self.screen_buffer = [
			np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
			np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
		]

		self.game_over = False
		self.lives = 0  # Will need to be set by reset().

	@property
	def observation_space(self):
		# Return the observation space adjusted to match the shape of the processed
		# observations.
		return gym.spaces.box.Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1), dtype=np.uint8)

	@property
	def action_space(self):
		return self.environment.action_space

	@property
	def reward_range(self):
		return self.environment.reward_range

	@property
	def metadata(self):
		return self.environment.metadata

	def reset(self):
		"""Resets the environment.

		Returns:
		observation: numpy array, the initial observation emitted by the
			environment.
		"""
		self.environment.reset()
		self.lives = self.environment.ale.lives()
		self._fetch_grayscale_observation(self.screen_buffer[0])
		self.screen_buffer[1].fill(0)
		return self._pool_and_resize()

	def render(self, mode):
		"""Renders the current screen, before preprocessing.

		This calls the Gym API's render() method.

		Args:
		mode: Mode argument for the environment's render() method.
			Valid values (str) are:
			'rgb_array': returns the raw ALE image.
			'human': renders to display via the Gym renderer.

		Returns:
		if mode='rgb_array': numpy array, the most recent screen.
		if mode='human': bool, whether the rendering was successful.
		"""
		return self.environment.render(mode)

	def step(self, action):
		"""Applies the given action in the environment.

		Remarks:

		* If a terminal state (from life loss or episode end) is reached, this may
			execute fewer than self.frame_skip steps in the environment.
		* Furthermore, in this case the returned observation may not contain valid
			image data and should be ignored.

		Args:
		action: The action to be executed.

		Returns:
		observation: numpy array, the observation following the action.
		reward: float, the reward following the action.
		is_terminal: bool, whether the environment has reached a terminal state.
			This is true when a life is lost and terminal_on_life_loss, or when the
			episode is over.
		info: Gym API's info data structure.
		"""
		accumulated_reward = 0.

		for time_step in range(self.frame_skip):
			# We bypass the Gym observation altogether and directly fetch the
			# grayscale image from the ALE. This is a little faster.
			_, reward, game_over, info = self.environment.step(action)
			accumulated_reward += reward

			if self.terminal_on_life_loss:
				new_lives = self.environment.ale.lives()
				is_terminal = game_over or new_lives < self.lives
				self.lives = new_lives
			else:
				is_terminal = game_over

			if is_terminal:
				break
			# We max-pool over the last two frames, in grayscale.
			elif time_step >= self.frame_skip - 2:
				t = time_step - (self.frame_skip - 2)
				self._fetch_grayscale_observation(self.screen_buffer[t])

		# Pool the last two observations.
		observation = self._pool_and_resize()

		self.game_over = game_over
		return observation, accumulated_reward, is_terminal, info

	def _fetch_grayscale_observation(self, output):
		"""Returns the current observation in grayscale.

		The returned observation is stored in 'output'.

		Args:
		output: numpy array, screen buffer to hold the returned observation.

		Returns:
		observation: numpy array, the current observation in grayscale.
		"""
		self.environment.ale.getScreenGrayscale(output)
		return output

	def _pool_and_resize(self):
		"""Transforms two frames into a Nature DQN observation.

		For efficiency, the transformation is done in-place in self.screen_buffer.

		Returns:
		transformed_screen: numpy array, pooled, resized screen.
		"""
		# Pool if there are enough screens to do so.
		if self.frame_skip > 1:
			np.maximum(self.screen_buffer[0], self.screen_buffer[1],
					out=self.screen_buffer[0])

		transformed_image = cv2.resize( self.screen_buffer[0],
										(self.screen_size, self.screen_size),
										interpolation=cv2.INTER_AREA )
		int_image = np.asarray(transformed_image, dtype=np.uint8)
		return np.expand_dims(int_image, axis=2)



