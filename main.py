import gym
import sys
import torch

from args import get_args
from ppo import ppo
from network import net
from eval_policy import eval_policy

def train(env, hyperparams, actor_model, critic_model):

	print(f"Training", flush=True)

	model = ppo(policy_class=net, env=env, **hyperparams)

	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)

	elif actor_model != '' or critic_model != '':
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)

	else:
		print(f"Training from scratch.", flush=True)

	model.learn(t_tot=200000000)

def test(env, actor_model):

	print(f"Testing {actor_model}", flush=True)

	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	policy = net(obs_dim, act_dim)
	policy.load_state_dict(torch.load(actor_model))

	eval_policy(policy=policy, env=env, render=True)

def main(args):

	hyperparams = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2
			  }

	env = gym.make('Pendulum-v0')

	if args.mode == 'train':
		train(env=env, hyperparams=hyperparams, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args()
	main(args)