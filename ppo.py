import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import time
import gym
import numpy as np

class ppo:
	def __init__(self, policy_class, env, **hyperparams):
		# Ensure the env is compatible with the code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters
		self.init_hyper(hyperparams)

		# Extract env info
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		# Initialize actor (policy) & critic (value function) policy_classworks
		self.actor = policy_class(self.obs_dim, self.act_dim)
		self.critic = policy_class(self.obs_dim, 1)

		# Define optimizers
		self.optim_actor = Adam(self.actor.parameters(), lr=self.lr)
		self.optim_critic = Adam(self.critic.parameters(), lr=self.lr)

		# Create the covariance matrix
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		self.logger = {
			't': 0,
			'i': 0,
			'len_batch': [],
			'rew_batch': [],
			'loss_actor': []
		}


	def learn(self, t_tot):
		print(f"Learning... Running {self.t_epis} timesteps per episode, ", end='')
		print(f"{self.t_batch} timesteps per batch for a total of {t_tot} timesteps")

		t = 0
		i = 0

		while t < t_tot:

			# Collect batch simulations
			obs_batch, act_batch, log_prob_batch, rtg_batch, len_batch = self.sample()

			t += np.sum(len_batch)
			i += 1

			self.logger['t'] = t
			self.logger['i'] = i

			# Calculate & Normalize advantage at k-th iteration
			V, _ = self.evaluate(obs_batch, act_batch)
			A_k = rtg_batch - V.detach()
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# Update the network for some n epochs
			for _ in range(self.epoch_per_iter):

				# Calculate ratios
				V, log_prob_cur = self.evaluate(obs_batch, act_batch)
				ratio = torch.exp(log_prob_cur - log_prob_batch)

				# Calculate surrogate losses
				sur1 = ratio * A_k
				sur2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * A_k

				# Calculate losses
				loss_actor = (-torch.min(sur1, sur2)).mean()
				loss_critic = nn.MSELoss()(V, rtg_batch)

				# Calculate gradients & perform backprop
				self.optim_actor.zero_grad()
				loss_actor.backward(retain_graph=True)
				self.optim_actor.step()

				self.optim_critic.zero_grad()
				loss_critic.backward()
				self.optim_critic.step()

				self.logger['loss_actor'].append(loss_actor.detach())

			self.log_summ()

			if i % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')


	def sample(self):
		# Collect batch of data from simulation
		obs_batch = []
		act_batch = []
		log_prob_batch = []
		rew_batch = []
		rtg_batch = [] # Reward to go
		len_batch = [] # Episodic lengths

		# Timesteps simulated so far
		t = 0
		while t < self.t_batch:

			rew_epis = []

			obs = self.env.reset()
			done = False

			for t_ep in range(self.t_epis):

				if self.render:
					self.env.render()

				t += 1

				obs_batch.append(obs)

				action, log_prob = self.get_action(obs)
				obs, rew, done, _ = self.env.step(action)

				rew_epis.append(rew)
				act_batch.append(action)
				log_prob_batch.append(log_prob)

				if done:
					break

			len_batch.append(t_ep + 1)
			rew_batch.append(rew_epis)

		obs_batch = torch.tensor(obs_batch, dtype=torch.float)
		act_batch = torch.tensor(act_batch, dtype=torch.float)
		log_prob_batch = torch.tensor(log_prob_batch, dtype=torch.float)

		rtg_batch = self.compute_rtg(rew_batch)

		self.logger['rew_batch'] = rew_batch
		self.logger['len_batch'] = len_batch

		return obs_batch, act_batch, log_prob_batch, rtg_batch, len_batch

	def compute_rtg(self, rew_batch):
		rtg_batch = []
		for rew_epis in reversed(rew_batch):
			# Discounted reward so far
			rew_dis = 0 
			for rew in reversed(rew_epis):
				rew_dis = rew + rew_dis * self.gamma
				rtg_batch.insert(0, rew_dis)
		rtg_batch = torch.tensor(rtg_batch, dtype=torch.float)

		return rtg_batch

	def get_action(self, obs):
		# Query the actor network for a mean action
		mean = self.actor(obs)
		dist = MultivariateNormal(mean, self.cov_mat)

		action = dist.sample()
		log_prob = dist.log_prob(action)

		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, obs_batch, act_batch):
		# Query critic network for V_obs for obs in obs_batch
		V = self.critic(obs_batch).squeeze()

		mean = self.actor(obs_batch)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_prob = dist.log_prob(act_batch)

		return V, log_prob

	def init_hyper(self, hyperparams):
		self.t_batch = 4800
		self.t_epis = 1600
		self.gamma = 0.95
		self.epoch_per_iter = 5
		self.clip = 0.2
		self.lr = 0.005

		self.render = False
		self.save_freq = 10
		self.seed = None

		for param, val in hyperparams.items():
			exec('self.' + param + ' = ' + str(val))

		if self.seed != None:
			assert(type(self.seed) == int)

			torch.manual_seed(self.seed)
			print(f'Success: Set seed to {self.seed}')

	def log_summ(self):
		t = self.logger['t']
		i = self.logger['i']
		avg_ep_len = np.mean(self.logger['len_batch'])
		avg_ep_rew = np.mean([np.sum(rew_ep) for rew_ep in self.logger['rew_batch']])
		avg_actor_loss = np.mean([loss.float().mean() for loss in self.logger['loss_actor']])

		avg_ep_len = str(round(avg_ep_len, 2))
		avg_ep_rew = str(round(avg_ep_rew, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		print(flush=True)
		print(f"-------------------- Iteration #{i} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_len}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rew}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		self.logger['len_batch'] = []
		self.logger['rew_batch'] = []
		self.logger['loss_actor'] = []