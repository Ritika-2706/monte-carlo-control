# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a grid world problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the grid world.

### States:
5 Terminal States:
G (Goal): The state the agent aims to reach.
H (Hole): A hazardous state that the agent must avoid at all costs.

11 Non-terminal States:
S (Starting state): The initial position of the agent.
Intermediate states: Grid cells forming a layout that the agent must traverse.

### Actions:
The agent has 4 possible actions:

0: Left
1: Down
2: Right
3: Up

### Transition Probabilities:
Slippery surface with a 33.3% chance of moving as intended and a 66.6% chance of moving in orthogonal directions. For example, if the agent intends to move left, there is a

33.3% chance of moving left, a
33.3% chance of moving down, and a
33.3% chance of moving up.

### Rewards:
The agent receives a reward of 1 for reaching the goal state and 0 otherwise.

## MONTE CARLO CONTROL ALGORITHM
Arbitrarily initialize the state value function V(s) and the policy π(s).

Generate an episode using π(s) and store the state, action, and reward sequence.

For each state s appearing in the episode:

G ← return following the first occurrence of s Append G to Returns(s) V(s) ← average(Returns(s)) For each state s in the episode: π(s) ← argmax_a ∑_s' P(s'|s,a)V(s') Repeat steps 2-4 until the policy converges.

Use the function decay_schedule to decay the value of epsilon and alpha.

Use the function gen_traj to generate a trajectory.

Use the function tqdm to display the progress bar.

After the policy converges, use the function np.argmax to find the optimal policy. The function takes the following arguments:

Q: The Q-table. axis: The axis along which the maximum value is found.

## MONTE CARLO CONTROL FUNCTION
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
![I1](https://github.com/user-attachments/assets/a36fabad-a6f3-4e07-866e-53251887742b)
![I2](https://github.com/user-attachments/assets/07f964db-372e-4693-96d1-e22b68e0d7ad)


## RESULT:
Thus a Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm.
