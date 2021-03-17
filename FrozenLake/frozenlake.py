from qAgent import QAgent
import gym
import argparse


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-environment', '--env', type=str, default='FrozenLake-v0')
parser.add_argument('-load','--l', type=bool, default=True)
parser.add_argument('-save', '--s', type=bool, default=True)
parser.add_argument('-episodes', '--ep', type=int, default=5000)
args = parser.parse_args()

NAME = args.env

# Create environment
env = gym.make(NAME)

# Create agent
agent = QAgent(NAME, n_actions=env.action_space.n, input_dims=env.observation_space.n)

# Boolean values
SAVE = args.s
LOAD = args.l
FILE_EXIST = agent.isExist

# Check file existance and load value
if LOAD and FILE_EXIST:
	agent.load_table()

n_episodes = args.ep
scores = []

# Episode loop
for i in range(n_episodes):
	observation = env.reset()
	score = 0
	done = False

	# Game loop
	while not done:
		# Choose action
		action = agent.choose_action(observation, i)
		
		# Receive new state, reward value, done
		observation_, reward, done, info = env.step(action)

		# Update table based on observations and reward
		agent.update_table(observation, action, reward, observation_)

		# Set new state as state
		observation = observation_

		# Add reward to score
		score += reward

	# Append list with total episode score 
	scores.append(score)

# Print last 50 episodes' scores
print(scores[-50:])

# Save q-table if its wanted
if SAVE:
	agent.save_table()
