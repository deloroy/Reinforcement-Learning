from imports import *

class Option():

    #Defining an option

    def __init__(self, env, concept, trajectories, size_option = 10, gamma_option = 0.95):
        # intialize option of reaching concept c : input set, terminal states, uncomputed Qlearning matrix
        # hence, one should call option.init_policy(trajectories) after this initialization
        # in order to compute its policy from a set of trajectories

        # input set of option
        self.input_set=set()
        for trajectory in trajectories:
            found_concept = False
            for (t,step) in enumerate(trajectory):
                if found_concept: break
                state = step[0]
                if state == concept:
                    found_concept = True
                    for s in range(max(0,t-size_option),t):
                        self.input_set.add(trajectory[s][0])

        # terminal state of option (a concept is a state)
        self.terminal_states = [concept]

        # initialize QLearning matrix of option
        self.Q = - math.inf * np.ones((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in env.state_actions[s]:
                self.Q[s, a] = np.random.rand()  # random initialisation of the Q function

        self.nb_visits = np.zeros((env.n_states, env.n_actions))

        self.policy = self.Q.argmax(axis=1)

        self.gamma_option = gamma_option
        self.size_option = size_option


    def init_policy(self, trajectories):
        #computes the policy of an option

        #building dataset of samples from all trajectories (experience replay)
        #https: // datascience.stackexchange.com / questions / 20535 / what - is -experience - replay - and -what - are - its - benefits

        dataset = []
        for trajectory in trajectories:
            for (t,step) in enumerate(trajectory[:-1]):
                state = step[0]
                if state in self.terminal_states:
                    for s in range(max(0,t-self.size_option),t):
                        step2 = trajectory[s]
                        dataset.append([step2[0],step2[1],trajectory[s+1][0]])
        dataset = np.array(dataset)

        # shuffling samples

        num_samples = dataset.shape[0]
        perm = np.random.permutation(num_samples)
        dataset = dataset[perm,:]

        # updating Q matrix of option

        for t in range(num_samples):

            state = int(dataset[t, 0])
            action =  int(dataset[t, 1])
            next_state = int(dataset[t, 2])

            reward_option = self.reward_option(next_state)

            learned_value = reward_option + self.gamma_option * self.Q[next_state, :].max()

            learning_rate = 1 / (self.nb_visits[state, action] + 1)
            self.Q[state, action] = (1 - learning_rate) * self.Q[state, action] + learning_rate * learned_value
            self.nb_visits[state,action] +=1

        # updating policy of option

        self.policy = self.Q.argmax(axis=1)


    def take_action(self, state):
        #returns action to take following the policy from state

        if state in self.input_set:
            return self.policy[state]
        else:
            print("State not in the input set of option. Can't take action.")
            exit()


    def reward_option(self, next_state):
        #returs the reward of the option for reaching next_state

        if next_state in self.terminal_states:
            return 10
        elif not(next_state in self.input_set):
            return -1
        else:
            return -1



class MacroQlearning():


    def __init__(self, env):
        #is defined by :
        #a (state,action/option) value matrix Q
        #a correspondance between the column index in Q matrix and the definition of the corresponding option
        #a (state,action/option) number of visits matrix nb_visits
        #the available actions from a state s (actions+options)

        self.num_primitive_actions = env.n_actions #num of primitive actions
        self.num_actions = self.num_primitive_actions #num of primitive actions + num of options

        # Q-learning matrix, Value function and Policy
        self.Q = - math.inf * np.ones((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in env.state_actions[s]:
                self.Q[s, a] = np.random.rand()  # random initialisation of the Q function
        self.nb_visits = np.zeros((env.n_states, env.n_actions)) #number of visits of the pair (state, action) for adaptive learning rate

        self.V = self.Q.max(axis=1)
        self.policy = self.Q.argmax(axis=1)

        # Correspondance between column index in Q matrix and definition of option
        self.idOption2Option = {}
        self.idOption2OptionTerminalState = {}

        # Redefining the possible options from state s : actions + options
        self.available_actions = {}
        for s in range(env.n_states):
            self.available_actions[s] = env.state_actions[s].copy()


    def add_option(self, option):
        #adds option : updates available actions from state,
        #Q matrix and nb_visits matrix, correspondance between column of Qmatrix and new option definition

        id_option = self.num_actions

        self.idOption2Option[id_option] = option
        self.idOption2OptionTerminalState[id_option] = option.terminal_states

        for s in option.input_set:
            self.available_actions[s].append(id_option)

        self.num_actions += 1

        num_states = self.Q.shape[0]
        Q_option = - math.inf * np.ones((num_states, 1))
        for s in option.input_set:
            Q_option[s,0] = np.random.rand()
        self.Q = np.hstack([self.Q,Q_option])

        nb_visits_option = np.zeros((num_states, 1))
        self.nb_visits = np.hstack([self.nb_visits, nb_visits_option])


    def learn(self, env, eps, nbEpisods, Tmax, plot=False, return_trajectories=True, verbose_option=False):
        # learn using MacroQLearning with the current primitive actions and options
        # env environment
        # eps tradeoff exploration/exploitation of the eps-greedy policy
        # nbEpisods number of episods
        # Tmax maximal length of a trajectory
        # (new episods if Tmax transitions or if we reach the goal)
        # plot : visualize average of discounted cumulated rewards across past episods
        # if return trajectories False : returns the list of the discounted cumulated rewards for each episod
        # if return trajectories True : return the list of trajectories stored as lists of triplets (states,actions,rewards)
        # and the list of the discounted cumulated rewards for each episod

        reward_list = []  # to check the convergence : store cumulated rewards over an episode

        if return_trajectories:
            trajectories = []

        for k in range(nbEpisods):

            state = env.reset()  # new initial state choice
            reward_episod = 0

            if return_trajectories:
                trajectory = []

            t = 0
            term = False  # reached absorbing state

            while ((t < Tmax) and not (term)):

                # Action choice
                u = np.random.uniform(0, 1)
                if u < eps:
                    action = self.Q[state, :].argmax()  # exploit the learned value
                else:
                    actions = self.available_actions[state] # env.state_actions[state]
                    action = np.random.choice(actions)  # explore action space


                if action < self.num_primitive_actions:


                    #*** ACTION IS A PRIMITIVE ACTION***

                    # Simulating next state and reward
                    nexts, reward, term = env.step(state, action)

                    if return_trajectories:
                        trajectory.append([state, action, reward])

                    # Updating the value of Q
                    learned_value = reward + env.gamma * self.Q[nexts, :].max()
                    learning_rate = 1 / (self.nb_visits[state, action] + 1)
                    self.Q[state, action] = (1 - learning_rate) * self.Q[state, action] + learning_rate * learned_value

                    # Updating reward
                    reward_episod += env.gamma ** t * reward

                    # Updating current state
                    state = nexts
                    t += 1

                    # Updating nb visits
                    self.nb_visits[state, action] += 1


                else:

                    #*** ACTION IS AN OPTION***

                    state_0 = state

                    id_option = action
                    option = self.idOption2Option[action]

                    if verbose_option:
                        print("\nOption taken. Terminal states : ")
                        for s in option.terminal_states: env.describe_state(s)

                    reward_option = 0
                    t_option = 0

                    while (not(state in option.terminal_states)) \
                            and (state in option.input_set):
                        #while subgoal not reached
                        #and still in input set of option

                        if (t > Tmax) or (term):
                            # time of episod > Tmax
                            # or we did not reach the big goal (absorbing state)
                            break

                        # Following option
                        action = option.take_action(state)

                        # Simulating next state and reward
                        nexts, reward, term = env.step(state, action)

                        if return_trajectories:
                            trajectory.append([state, action, reward])

                        # Updating reward
                        reward_episod += env.gamma ** t * reward

                        # Updating reward option
                        reward_option += env.gamma ** t_option * reward

                        # Updating current state
                        state = nexts
                        t += 1
                        t_option += 1

                    if verbose_option:
                        print("Option ended ")
                        if state in option.terminal_states:
                            print("because option goal reached")
                        elif not(state in option.input_set):
                            print("because option input set left")

                    # Updating the value of Q
                    # cf. Sutton article on MACROS
                    learned_value = reward_option + env.gamma**t_option * self.Q[state, :].max()
                    learning_rate = 1 / (self.nb_visits[state, id_option] + 1)

                    self.Q[state_0, id_option] = (1 - learning_rate) * self.Q[state_0, id_option] + learning_rate * learned_value

                    # Updating nb visits
                    self.nb_visits[state_0, id_option] += 1

            reward_list.append(reward_episod)

            #print("############# ",t)

            if return_trajectories:
                trajectories.append(trajectory)

        # compute the value function and the policy
        self.V = self.Q.max(axis=1)
        self.policy = self.Q.argmax(axis=1)

        if (plot):
            # Convergence check
            reward_list_2 = np.cumsum(np.array(reward_list)) / (np.arange(nbEpisods) + 1)
            plt.scatter(list(range(nbEpisods)), reward_list_2)
            plt.xlabel("Episodes")
            plt.ylabel("Average of cumulated discounted reward across past episodes")
            plt.title("Convergence check")
            plt.show()

        if return_trajectories:
            return trajectories, reward_list
        else:
            return reward_list
