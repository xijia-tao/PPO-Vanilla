# Proximal Policy Optimization (PPO) - Vanilla Ver.

The repository contains a simple implementation of PPO using PyTorch, based on the pseudocode provided in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

> Reference: [The Medium Series by Eric Yu](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)

## Navigate Through the Files

- `args.py`:  A helper function that helps us parse the arguments. It's where you can choose mode (`train`/`test`), specify the directory of models to be tested / continued training if any. 
- `eval_policy.py`: A module that evaluates an existing policy (i.e., actor model).
- `main.py`:  The interface where we can define the hyperparameters and kick start our training / testing.
- `network.py`: The class defined for both actor and critic networks. Here I used a simple fully connected NN consisting of 3 linear layers.
- `ppo.py`:  The `ppo` class where all the learning takes place, the heart of the PPO algorithm. It follows the pseudocode completely except the addition of a (rather common) technique of *normalization*, which decreases the variance of advantages and results in more stable and faster convergence.
- `ppo_actor.pth`/ `ppo_critic.pth`: The files where I saved the model parameters. If you wish get a feel of the trained model, opt for the relevant options in `args.py`. Otherwise when you run `main.py`, both files would be overrided with the new parameters while training.

![](/assets/pseudocode.png)

## Usage

In short:

- `python main.py` from command line if you wish to train the agent from scratch.
- Change `train` to `test` and specify the directory of your actor network (`ppo_actor.pth` by default) in `args.py` if you wish to evaluate the performance of your agent visually. Then run `main.py`
- Specify the directory of both your actor and policy networks if you wish to continue training an existing PPO model. Then run `main.py`.

## Results

![](/assets/train.png)

The average episodic return converges to approx. -200 after 3,000,000 timesteps. The performance is comparable to the stable baseline of PPO2 given sufficient number of iterations.

![Demo: RL Agent on Pendulum-v0 After Training](/assets/ppo_pendulum.gif)