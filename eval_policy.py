def log_summary(len_ep, rew_ep, num_ep):
	len_ep = str(round(len_ep, 2))
	rew_ep = str(round(rew_ep, 2))

	print(flush=True)
	print(f"-------------------- Episode #{num_ep} --------------------", flush=True)
	print(f"Episodic Length: {len_ep}", flush=True)
	print(f"Episodic Return: {rew_ep}", flush=True)
	print(f"------------------------------------------------------", flush=True)
	print(flush=True)

def sample(policy, env, render):
	while True:
		obs = env.reset()
		done = False

		t = 0

		len_ep = 0           
		rew_ep = 0      

		while not done:
			t += 1

			if render:
				env.render()

			action = policy(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)

			rew_ep += rew
			
		len_ep = t

		yield len_ep, rew_ep

def eval_policy(policy, env, render=False):
	for num_ep, (len_ep, rew_ep) in enumerate(sample(policy, env, render)):
		log_summary(len_ep=len_ep, rew_ep=rew_ep, num_ep=num_ep)