from DiverseDensity import *
from gridworld import *
from Qlearning import *
import itertools
import pandas as pd

# Environment definition and initialisation

env = GridWorld_SIMPLIFIED_PB
env.reset()

# Parameters

Tmax = 1000  # time horizon
nbEpisodsTotal = 500 # nb of episods
eps_0 = 0.4  # initial exploration/exploitation tradeoff

size_option = 5  # n_actions before reaching a concept considered for option input set
gamma_option = 0.95  # discounting factor option, to compute option own's policy

lambd = 0.9 # for running average of how often a concept appears as a peak
#print("lamb/(1-lambd) : ",lambd/(1-lambd)) #convergence value for early persisting peak
threshold_peak = 0.9999 * lambd/(1-lambd) #threshold on the runnning average to consider a peak concept as a subgoal candidate


N_runs = 30 #number of runs of DDMQ (and FlatQlearning) for experimental comparison
window_moving_avg = 20 #size of the moving average over episods for plot of the number of steps to reach the goal throughout learning

#####################
#####################
#####################
#list of taboo concepts (too close to the goal of the task to be considered as subgoal candidate)

taboo_concepts = set()
size_window = list(range(3))
window = [(i, j) for i, j in itertools.product(size_window,size_window) if ((i+j >=0) and (i + j <= 2))] #voisinage
for destination in env.idx_destination_ranks:
    pos_dest = env.positionId2coord[env.idxDestination2positionId[destination]]
    for w in window:
        try:
            pos = env.coord2positionId[pos_dest[0]+w[0], pos_dest[1]+w[1]]
            taboo_concepts.add(env.state2Id[pos,1,0])
        except:
            pass
taboo_concepts = list(taboo_concepts)

#####################
#####################
#####################

def DDMQ(plot=False):
    #performs one run of DDMQ with the selected parameters
    #returns the Macro QLearning class at the end of the learning,
    #and the list of trajectories stored as lists of triplets (states,actions,rewards)
    #and the list of the discounted cumulated rewards for each episod
    #and the running average of how often each concept appears as a peak at the end of episode 50
    #plot : visualize average of discounted cumulated rewards across past episods

    # running average of how often a concept appears as a peak at the end of episode 50
    score_concepts_50 = None

    all_trajectories = [] #all trajectories
    all_bags = {"pos_bags":[], "neg_bags":[]} #all bags
    rewards = [] #cumulated discounted reward per trajectory

    # a concept is reaching an individual state
    proba_concepts = np.ones(env.n_states) #probability of each concept given the bags
    score_concepts = np.zeros(env.n_states) #running average of how often a concept appears as a peak

    MQL = MacroQlearning(env)

    eps=eps_0 #intial exploration/exploitation tradeoff

    for k in range(nbEpisodsTotal):

        #print("\n#########################################")
        #print("#########################################")
        print("trial #"+str(k))

        #############################################

        #print("A) Macro-QLearning")

        #learn and collect trajectories
        trajectory, reward = MQL.learn(env, eps, 1, Tmax, verbose_option = True, plot = False, return_trajectories=True)

        all_trajectories += trajectory
        rewards += reward

        ############################################

        #print("B) Look for DD peaks")

        bags = create_bags_from_trajectory(trajectory,Tmax)
        bags = {"pos_bags":bags["pos_bags"]}#we only create positive bags
        all_bags["pos_bags"] += bags["pos_bags"]
        #print("\nnumber of successful trajectories : ", len(bags["pos_bags"]))

        if len(all_bags["pos_bags"]) > 0:  # there is at least one positive bag

            #update proability of concepts given bags, and compute new peak concepts
            proba_concepts, peak_concepts = update_DD_peaks(env, proba_concepts, bags)
            for concept in peak_concepts:
                score_concepts[concept] += 1
                score_concepts[concept] *= lambd

            #save score of concepts at the end of episode 50
            if k==50:
                score_concepts_50 = score_concepts.copy()

            #select subgoal candidates among peak concepts
            selected_concepts = []
            for concept in peak_concepts:
                if score_concepts[concept] > threshold_peak:
                    if not(concept in taboo_concepts): #admissible states : not too close to each start or end of all succ traj
                        selected_concepts.append(concept)

            #select chosen subgoal candidate, and create option (max one per run)
            if len(selected_concepts)>0:
                chosen_concept = np.random.choice(selected_concepts)

                # trajectories that will be mined to create the option
                if len(all_trajectories) < 100:
                    memory = all_trajectories
                else:
                    memory = all_trajectories[-100:] #last trajectories

                option = Option(env, chosen_concept, memory, size_option =size_option, gamma_option=gamma_option)
                option.init_policy(memory) #compute option policy with experience replay
                MQL.add_option(option)
                print("\nOption created. Terminal states : ")
                for s in option.terminal_states: env.describe_state(s)

        if k%100==0:
            eps = max(0.9, eps+0.1) # Increase eps to favor exploitation


    if (plot):
        # Convergence check
        rewards_2 = np.cumsum(np.array(rewards)) / (np.arange(nbEpisodsTotal) + 1)
        plt.scatter(list(range(nbEpisodsTotal)), rewards_2)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulated discounted reward")
        plt.title("Convergence check, nbEpisods per DD iteration : "+str(nbEpisodsTotal))
        plt.show()

    return MQL, all_trajectories, rewards, score_concepts_50




###################
###################
###################
#LEARNING WITH MACRO-QLEARNING AND VIZ TRAJECTORIES

print("MACRO Q-LEARNING")
MQL, traj_with_option, reward_with_option, score_concepts_50 = DDMQ(plot=False)
#print("final V",V)
#print("final policy : ", policy)

#Trajectory visualisation
print("#####################")
print('Visualise trajectory')

TmaxDemo = 30

for t in range(0):

    env.render = True
    state = env.reset()
    fps = 5

    i = 0
    while i<TmaxDemo:

        #print("state : ")
        #env.describe_state(state)

        action = MQL.policy[state]
        term = False

        if action<MQL.num_primitive_actions:
            print("action : ", env.action_names[action])
            nexts, reward, term = env.step(state, action)
            state = nexts
            i+=1
            time.sleep(1. / fps)

            #print("state : ")
            #env.describe_state(state)

        else:
            id_option = action
            option = MQL.idOption2Option[action]
            print(colored("option with terminal state : ","yellow"))
            for s in option.terminal_states: env.describe_state(s)

            while (not(state in option.terminal_states)) \
                    and (state in option.input_set):
                # while subgoal not reached
                # and still in input set of option

                if (i ==TmaxDemo) or (term):
                    # time of episod > Tmax
                    # or we did not reach the big goal (absorbing state)
                    break

                # Following option
                action = option.take_action(state)
                print("action : ", env.action_names[action])
                nexts, reward, term = env.step(state, action)
                state = nexts
                i += 1
                time.sleep(1. / fps)

                #print("state : ")
                #env.describe_state(state)


        if term:
            env.render = False
            break


###################
###################
###################
#COMPARISON BETWEEN DDMQ AND BASIC FLAT QLEARNING

reward_no_option = np.zeros((nbEpisodsTotal))
n_steps_no_option = np.zeros((nbEpisodsTotal))

reward_option = np.zeros((nbEpisodsTotal))
n_steps_option = np.zeros((nbEpisodsTotal))

score_concepts_50_mean = np.zeros(env.n_states)

for k in range(N_runs):

    print("BASIC Q-LEARNING")
    print('Visualise cumulated discounted reward from the first trajectory')
    Q, V, policy, traj_without_option, reward_without_option = Qlearning(env, eps_0, nbEpisodsTotal, Tmax, plot=False, return_trajectories=True)

    print("MACRO Q-LEARNING")
    MQL, traj_with_option, reward_with_option, score_concepts_50 = DDMQ(plot=False)

    reward_no_option += np.array(reward_without_option)
    n_steps_no_option += np.array([len(t) for t in traj_without_option])
    reward_option += np.array(reward_with_option)
    n_steps_option += np.array([len(t) for t in traj_with_option])

    score_concepts_50_mean += score_concepts_50

reward_no_option /= N_runs
reward_option /= N_runs
n_steps_no_option /= N_runs
n_steps_option /= N_runs

#plot 1 : how often a concept appeared as a peak at the end of episode 50 ?
score_concepts_50_mean /= N_runs
plot_score_concepts(env,score_concepts_50_mean,taboo_concepts)

#plot 2 : average of discounted cumulated rewards across past episodes
plt.plot(list(range(nbEpisodsTotal)), np.cumsum(reward_no_option)/ (np.arange(nbEpisodsTotal) + 1), color="red", label="no option")
plt.plot(list(range(nbEpisodsTotal)), np.cumsum(reward_option)/ (np.arange(nbEpisodsTotal) + 1) , color="blue", label="with option")
plt.xlabel("Episodes")
plt.ylabel("Average of discounted cumulated rewards across past episodes")
plt.title("Convergence check")
plt.legend()
plt.show()

#plot 3 : evolution of number of steps to reach the goal throughout learning
episods = list(range(nbEpisodsTotal))
df_no_option = pd.DataFrame({"n_steps":n_steps_no_option})
df_option = pd.DataFrame({"n_steps":n_steps_option})

to_plot_option = df_option[["n_steps"]].rolling(window=window_moving_avg).mean()
to_plot_no_option = df_no_option[["n_steps"]].rolling(window=window_moving_avg).mean()

ax0 = to_plot_option.plot(color="blue")
to_plot_no_option.plot(ax=ax0,color="red")
ax0.legend(["with option","no option"])

title = "Evolution of number of steps to reach the goal"

#if window_moving_avg > 0:
#    title = title + " (moving average over " + str(window_moving_avg) + " episods)"
plt.xlabel("Episodes")
plt.ylabel("n_steps")
plt.title(title)
plt.show()


