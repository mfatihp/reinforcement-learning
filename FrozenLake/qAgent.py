import numpy as np
import os


# Agent Class
class QAgent(object):
	def __init__(self, name, n_actions, input_dims, learning_rate=1e-2, gamma=0.99, table_dir='qtables'):
		"""
		name          : required for saving
		n_actions     : Discrete action space number
		input_dims    : State space number
		learning_rate : For Q-table value update
		gamma         : Discount factor
		table_dir     : Table backup location
		"""
		self.input_dims = input_dims
		self.n_actions = n_actions
		self.gamma = gamma
		self.lr = learning_rate

		self.name = name
		self.table_dir = table_dir

		self.file_name = os.path.join(table_dir, name)

		# Check table file existance
		self.isExist = os.path.isfile(self.file_name + '.npy')

		# Action list
		self.action_space = [i for i in range(n_actions)]

		# Q-table matrix filled with zeros
		self.q_table = np.zeros((self.input_dims, self.n_actions), dtype=np.float32)


	def update_table(self, state, action, reward, new_state):
		# Compute new q value and raplace old one
		self.q_val = self.q_table[state, action] * (1 - self.lr) + self.lr * (reward + self.gamma * np.max(self.q_table[new_state,:])) 
		self.q_table[state, action] = self.q_val


	def choose_action(self, state, episode):
		# Choose a float number between 0-1 randomly
		rand = np.random.uniform(0, 1)

		# Calculate exploration value
		self.eps = 0.01 + (0.99 - 0.01) * np.exp(-0.001 * episode)

		# Exploration value comparison
		if rand < self.eps:
			# Explore
			action = np.random.choice(self.action_space)
		else:
			# Remember best action
			action = np.argmax(self.q_table[state])

		return action


	def save_table(self):
		# Save q-table as numpy array
		np.save(self.file_name, self.q_table)


	def load_table(self):
		# Load q-table
		self.q_table = np.load(self.file_name + ".npy")
