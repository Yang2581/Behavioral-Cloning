import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer
import gym
from matplotlib import pyplot as plt

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.9,
                 max_size=5000, fc1_dims=300, fc2_dims=200,
                 batch_size=32):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval() #启动评估模式，关闭dropout和batchnorm
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        self.actor.train() #启动训练模式，开启dropout....

        return mu.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self, path = 'tmp/ddpg'):
        self.actor.save_checkpoint(chkpt_dir=path)
        self.target_actor.save_checkpoint(chkpt_dir=path)
        self.critic.save_checkpoint(chkpt_dir=path)
        self.target_critic.save_checkpoint(chkpt_dir=path)

    def load_models(self, path = 'tmp/ddpg'):
        self.actor.load_checkpoint(chkpt_dir=path)
        self.target_actor.load_checkpoint(chkpt_dir=path)
        self.critic.load_checkpoint(chkpt_dir=path)
        self.target_critic.load_checkpoint(chkpt_dir=path)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

if __name__ == '__main__': # gym环境下的测试代码，并产生数据
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=3, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300,
                    n_actions=env.action_space.shape[0])

    n_games = 300
    agent.load_models()
    best_score = env.reward_range[0]
    print(best_score)
    score_history = []
    dataset = {"observations":[], "actions":[]}
    idx = 0

    for i in range(n_games):

        observation = env.reset()
        done = False
        score = 0
        # agent.noise.reset()
        while not done:
            env.render()
            dataset['observations'].append(observation)
            action = agent.choose_action(observation)
            dataset['actions'].append(action)

            observation_, reward, done, info = env.step([action])

            agent.remember(observation, action, reward, observation_, done)
            # agent.learn()
            score += reward
            observation = observation_

            idx += 1
            print("idx= " + str(idx))
            if idx > 10000:
                dataset["observations"] = np.array(dataset["observations"]).reshape(-1, 3)
                dataset["actions"] = np.array(dataset["actions"]).reshape(-1, 1)

                np.save('datasets10000.npy', dataset)
                break
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if idx > 10000:
            break
        # if score > avg_score:
        #     best_score = avg_score
        #     agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    # plt.plot(x, score_history)
    # plt.show()
