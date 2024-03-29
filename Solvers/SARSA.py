# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class Sarsa(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith('Discrete'), str(self) + \
                                                                  " cannot handle non-discrete state spaces"
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            self.options.steps: number of steps per episode
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # get action
        p_action = self.epsilon_greedy_action(state)
        action = np.random.choice(len(p_action), p=p_action)
        # for each step of the episode
        for t in range(self.options.steps):
            # take action
            next_state, reward, done, _ = self.step(action)

            # get next action
            p_action = self.epsilon_greedy_action(next_state)
            next_action = np.random.choice(len(p_action), p=p_action)

            # update q values with q-learning
            self.Q[state][action] += self.options.alpha * (reward + self.options.gamma * self.Q[next_state][next_action] - self.Q[state][action])

            # update state and action
            state = next_state
            action = next_action
            # check if not done
            if done:
                break

    def __str__(self):
        return "Sarsa"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(state):
            best_action = np.argmax(self.Q[state])
            return best_action

        return policy_fn

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: size of the action space
            np.argmax(self.Q[state]): action with highest q value

        Returns:
            Probability of taking actions

        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # get greedy action
        opt = np.argmax(self.Q[state])
        nA = self.env.action_space.n
        # build epsilon greedy policy
        # for optimal action p = epsilon/nA + 1 - epsilon
        # for non-optimal actions p = epsilon/na
        p = np.zeros(nA) + (self.options.epsilon / nA)
        p[opt] += 1 - self.options.epsilon

        return p

    def plot(self,stats):
        plotting.plot_episode_stats(stats)
