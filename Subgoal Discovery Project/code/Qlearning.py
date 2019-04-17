from imports import *

def Qlearning(env, eps, nbEpisods, Tmax, plot=False, plot_test=False, return_trajectories=False):

    #env environment

    #eps initial tradeoff exploration, exploitation of the eps-greedy policy
    #nbEpisods number of episods
    #Tmax maximal length of a trajectory 
    #(new episods if Tmax transitions or if we reach the goal)

    #plot :  visualize average of discounted cumulated rewards across past episods
    #plot_test : simulate trajectories following learnt policy at different steps of the learning
    #and plot the number needed of steps to reach the goal at these timesteps

    #if return trajectories is False: returns Q,V, policy
    #if return trajectories is True: return also the list of trajectories stored as lists of triplets (states,actions,rewards)
    #as well as the list of the discounted cumulated rewards for each episod

    if plot_test:
        timesteps_test = []
        mean_n_steps_test = []
        std_n_steps_test = []

    #Initialisation
    Q = - math.inf * np.ones((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in env.state_actions[s]:
            Q[s,a]=np.random.rand() # random initialisation of the Q function
                
    #nb_visits[s,a] number fo visits of the state-action (s,a) until current trajectory
    nb_visits = np.zeros((env.n_states, env.n_actions))
    
    reward_list=[]  # to check the convergence : store cumulated rewards over an episode

    if return_trajectories:
       trajectories = []
    
    for k in range(nbEpisods):
        
        state = env.reset()  # new initial state choice
        reward_episod=0

        if return_trajectories:
           trajectory = []

        t = 0
        term = False #reached absorbing state

        while ((t<Tmax) and not(term)): 
        
            # Action choice
            u = np.random.uniform(0,1)
            if u<eps:
                action = Q[state, :].argmax() # exploit the learned value
            else:
                action = np.random.choice(env.state_actions[state])  # explore action space

            # Simulating next state and reward
            nexts, reward, term = env.step(state, action)
  
            if return_trajectories:
               trajectory.append([state,action,reward])

            # Updating the value of Q
            learned_value = reward + env.gamma*Q[nexts,:].max()
            learning_rate = 1/(nb_visits[state,action]+1)
            Q[state, action] = (1-learning_rate)*Q[state, action] + learning_rate*learned_value

            # Updating reward
            reward_episod += env.gamma ** t * reward

             # Updating nb of visits
            nb_visits[state, action] += 1
            
            # Updating current state 
            state = nexts
            t+=1
            
        #Updating learning parameters
        #if k%2200==0:
        #     eps = max(0.9, eps+0.1) # Increase eps to favor exploitation
        if k%100==0:
            eps = max(0.9, eps+0.1) # Increase eps to favor exploitation

        reward_list.append(reward_episod)
  
        if return_trajectories:
           trajectories.append(trajectory)

        #################
        #################
        #################
        #################

        if plot_test:
            if ((k  % 5000 == 0)):
                timesteps_test.append(k)
                policy_tmp = Q.argmax(axis=1) #current policy
                len_traj = []
                for t in range(100):
                    env.render = False
                    state = env.reset()
                    i = 0
                    while (i<100):
                        action = policy_tmp[state]
                        nexts, reward, term_test = env.step(state, action)
                        state = nexts
                        i+=1
                        if term_test:
                            break
                    len_traj.append(i)
                mean_n_steps_test.append(np.mean(len_traj))
                std_n_steps_test.append(np.std(len_traj))

    #################
    #################
    #################

    if plot_test:
        mean_n_steps_test = np.array(mean_n_steps_test)
        std_n_steps_test = np.array(std_n_steps_test)
        plt.scatter(timesteps_test, mean_n_steps_test+std_n_steps_test,label="upper bound of the confidence interval")
        plt.scatter(timesteps_test,mean_n_steps_test,label="mean")
        plt.scatter(timesteps_test, mean_n_steps_test-std_n_steps_test,label="lower bound of the confidence interval")
        plt.xlabel("Episodes")
        plt.ylabel("Nb of steps to reach the goal")
        plt.legend()
        plt.show()

    #################
    #################
            
    # compute the value function and the policy
    V = Q.max(axis=1)
    policy = Q.argmax(axis=1)  
    
    if(plot):
        # Convergence check
        reward_list_2 = np.cumsum(np.array(reward_list)) / (np.arange(nbEpisods) + 1)
        plt.scatter(list(range(nbEpisods)), reward_list_2)
        plt.xlabel("Episodes")
        plt.ylabel("Average of cumulated discounted reward across past episods")
        plt.title("Convergence check")
        plt.show()

    if return_trajectories:
       return Q, V, policy, trajectories, reward_list
    else:
       return Q, V, policy



