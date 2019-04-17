from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time

env = GridWorld1

if False:
################################################################################
# investigate the structure of the environment
# - env.n_states: the number of states
# - env.state2coord: converts state number to coordinates (row, col)
# - env.coord2state: converts coordinates (row, col) into state number
# - env.action_names: converts action number [0,3] into a named action
# - env.state_actions: for each state stores the action availables
#   For example
#       print(env.state_actions[4]) -> [1,3]
#       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
# - env.gamma: discount factor
################################################################################
    print(env.state2coord)
    print(env.coord2state)
    print(env.state_actions)
    for i, el in enumerate(env.state_actions):
            print("s{}: {}".format(i, env.action_names[el]))

################################################################################
# Policy definition
# If you want to represent deterministic action you can just use the number of
# the action. Recall that in the terminal states only action 0 (right) is
# defined.
# In this case, you can use gui.renderpol to visualize the policy
################################################################################
    pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
    gui.render_policy(env, pol)

################################################################################
# Try to simulate a trajectory
# you can use env.step(s,a, render=True) to visualize the transition
################################################################################
    env.render = True
    state = 0
    fps = 1
    for i in range(5):
            action = np.random.choice(env.state_actions[state])
            nexts, reward, term = env.step(state,action)
            state = nexts
            time.sleep(1./fps)

################################################################################
# You can also visualize the q-function using render_q
################################################################################
# first get the maximum number of actions available
    max_act = max(map(len, env.state_actions))
    q = np.random.rand(env.n_states, max_act)
    gui.render_q(env, q)

#Questions are answered in the Jupyter notebook