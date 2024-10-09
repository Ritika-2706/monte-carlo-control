# MONTE CARLO CONTROL ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## MONTE CARLO CONTROL ALGORITHM
Include the steps involved in the Monte Carlo control algorithm

## MONTE CARLO CONTROL FUNCTION
Include the Monte Carlo control function

## PROGRAM
```
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi[state])
            steps += 1
        results.append(state == goal_state)
    return np.mean(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            action = pi[state]
            state, reward, done, _ = env.step(action)
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
#LEFT, RIGHT = range(2)
p

def decay_schedule(
    init_value, min_value, decay_ratio,
    max_steps, log_start = -2, log_base=10):


    decay_steps = int(max_steps * decay_ratio)
    values = np.logspace(log_start, 0, decay_steps, base=log_base)
    values = (values - values.min()) / (values.max() - values.min())
    values = init_value + (min_value - init_value) * values
    values = np.concatenate((values, np.full(max_steps - decay_steps, min_value)))
    return values

def generate_trajectory(select_action, Q, epsilon, env, max_steps=200):
    state, done, trajectory = env.reset(), False, []
    for _ in range(max_steps):
        if done:
            break
        action = select_action(state, Q, epsilon)
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    return np.array(trajectory, object)


def mc_control (env, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n


  Q = np.zeros((nS, nA))
  returns_count = np.zeros((nS, nA))
  alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
  epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

  def select_action(state, Q, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(nA)
        return np.argmax(Q[state])

  for episode in range(n_episodes):
        epsilon = epsilons[episode]
        alpha = alphas[episode]
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)

        G = 0
        visited = set()

        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = gamma * G + reward
            if (state, action) not in visited or not first_visit:
                returns_count[state, action] += 1
                Q[state, action] += alpha * (G - Q[state, action])
                visited.add((state, action))


  pi = np.argmax(Q, axis=1)
  V = np.max(Q, axis=1)


  #return Q, V, pi, Q_track, pi_track
  return Q, V, pi

optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes = 9000)
print('Name:Ritika S  Register Number: 212221240046')
print_state_value_function(optimal_Q, P, n_cols=4, prec=2, title='Action-value function:')
print_state_value_function(optimal_V, P, n_cols=4, prec=2, title='State-value function:')
print_policy(optimal_pi, P)

# Find the probability of success and the mean return of you your policy
print('Name: Ritika S Register Number: 212221240046')
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))
```


## OUTPUT:
### Name:
### Register Number:

Mention the Action value function, optimal value function, optimal policy, and success rate for the optimal policy.

## RESULT:

Write your result here
