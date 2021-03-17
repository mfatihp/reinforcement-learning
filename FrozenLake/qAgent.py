import numpy as np
import os



class QAgent(object):
	def __init__(self, name, n_actions, input_dims, learning_rate=1e-2, gamma=0.99, table_dir='qtables'):
		self.input_dims = input_dims
		self.n_actions = n_actions
		self.gamma = gamma
		self.lr = learning_rate

		self.name = name
		self.table_dir = table_dir

		self.file_name = os.path.join(table_dir, name)
		self.isExist = os.path.isfile(self.file_name + '.npy')

		self.action_space = [i for i in range(n_actions)]

		self.q_table = np.zeros((self.input_dims, self.n_actions), dtype=np.float32)


	def update_table(self, state, action, reward, new_state):
		self.q_val = self.q_table[state, action] * (1 - self.lr) + self.lr * (reward + self.gamma * np.max(self.q_table[new_state,:])) 
		self.q_table[state, action] = self.q_val


	def choose_action(self, state, episode):
		rand = np.random.uniform(0, 1)
		self.eps = 0.01 + (0.99 - 0.01) * np.exp(-0.001 * episode)

		if rand < self.eps:
			action = np.random.choice(self.action_space)
		else:
			action = np.argmax(self.q_table[state])

		self.eps = 0.01 + (0.99 - 0.01) * np.exp(-0.001 * episode)

		return action


	def save_table(self):
		np.save(self.file_name, self.q_table)


	def load_table(self):
		self.q_table = np.load(self.file_name + ".npy")
