from qAgent import QAgent
import gym
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-environment', '--env', type=str, default='FrozenLake-v0')
parser.add_argument('-load','--l', type=bool, default=True)
parser.add_argument('-save', '--s', type=bool, default=True)
parser.add_argument('-episodes', '--ep', type=int, default=5000)
args = parser.parse_args()

NAME = args.env
env = gym.make(NAME)

agent = QAgent(NAME, n_actions=env.action_space.n, input_dims=env.observation_space.n)

SAVE = args.s
LOAD = args.l
FILE_EXIST = agent.isExist

n_episodes = args.ep
scores = []

if LOAD and FILE_EXIST:
	agent.load_table()

for i in range(n_episodes):
	observation = env.reset()
	score = 0
	done = False
	while not done:
		action = agent.choose_action(observation, i)
		observation_, reward, done, info = env.step(action)
		agent.update_table(observation, action, reward, observation_)

		observation = observation_
		score += reward

	scores.append(score)
print(scores[-50:])

if SAVE:
	agent.save_table()
