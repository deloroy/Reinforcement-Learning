#Qlearning on TAXI problem

from Qlearning import *
from gridworld import *
import time

# Environment definition and initialisation

env = GridWorld_ORIGINAL_PB
#env = GridWorld_SIMPLIFIED_PB

env.reset()

# Parameters
Tmax=100             #Time horizon
nbEpisods=10000      #nb of episods
eps=0.4 #0.7         #initial exploration/exploitation tradeoff
    
#Q learning

print('Visualise cumulated discounted reward from the first trajectory')
Q, V, policy = Qlearning(env, eps, nbEpisods, Tmax, plot=True, return_trajectories=False)
print("final V",V)
print("final policy : ", policy)

#Visualise policy
print('Visualise policy')
gui.render_policy(env,policy)

#Trajectory visualisation
print('Visualise trajectory')

for t in range(10):
    env.render = True
    state = env.reset()
    fps = 10
    for i in range(20): #Tmax):

        action = policy[state]

        print("state : ")
        env.describe_state(state)

        nexts, reward, term = env.step(state, action)
        if term:
            env.render = False
            break
        state = nexts
        time.sleep(1./fps)
