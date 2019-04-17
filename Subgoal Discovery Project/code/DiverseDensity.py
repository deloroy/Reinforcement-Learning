from MacroQLearning import *
from gridworld import grid_SIMPLIFIED_PB

def create_bags_from_trajectory(trajectories, Tmax):
    #trajectories is a list of lists of triplets
    #each list of triplet is a trajectory with stored, at each time step, the state, the action, the reward
    #Tmax is the horizon of an episod
    #returnsa  dictionnary with states sequences as positive / negative bags
    #a succesfull trajetcory reaches the goal before the end of the episod Tmax
    pos_bags = []
    neg_bags = []
    for traj in trajectories:
        traj_states = [t[0] for t in traj]  #storing only states (not actions or rewards)
        if len(traj)<Tmax and traj[-1][2]==20: #absorbing state and corresponds to the goal (reward 20)
            pos_bags.append(traj_states)
        else:
            neg_bags.append(traj_states)
    return {"pos_bags":pos_bags,"neg_bags":neg_bags}


def similarity(env, id_state, id_concept):
    #return the similairy measure between a state and a concept
    #ie. between two states (a concept is reaching a state in a first approach)

    state_concept = env.id2State[id_concept]
    x_concept, y_concept = env.positionId2coord[state_concept[0]]
    pass_concept = state_concept[1]

    state = env.id2State[id_state]
    x_state, y_state = env.positionId2coord[state[0]]
    pass_state = state[1]

    return np.exp(-(x_concept - x_state)**2 -(y_concept - y_state)**2 -(pass_concept-pass_state)**2)


def proba_concept_given_bag(env, concept, bag, positive_bag =True):
    #returns the probaility of the concept given a bag
    res = 1
    for t in range(len(bag)):
        id_state = bag[t]
        res *= 1 - similarity(env, id_state, concept)

    if positive_bag:
        return 1-res
    else:
        return res

def DD_peaks(env,bags,plot=False):
    #returns the index of the peak concepts given a set of bags

    num_concepts =  env.n_states

    proba_concept = np.ones(num_concepts)

    for bag in bags["pos_bags"]:
        for concept in range(num_concepts):
            proba_concept[concept] *=  proba_concept_given_bag(env,concept, bag, positive_bag =True)

    if "neg_bags" in bags.keys():
        for bag in bags["neg_bags"]:
            for concept in range(num_concepts):
                proba_concept[concept] *=  proba_concept_given_bag(env, concept, bag, positive_bag = False)

    return np.argwhere(proba_concept == np.max(proba_concept)).flatten()


def update_DD_peaks(env,proba_concepts, bags):
    #update the pobability of the concepts proba_concepts
    #given the new bags "bags"
    #returns the new probabilities proba_concepts and the new peak concepts

    num_concepts =  env.n_states

    for bag in bags["pos_bags"]:
        for concept in range(num_concepts):
            proba_concepts[concept] *=  proba_concept_given_bag(env,concept, bag, positive_bag =True)

    if "neg_bags" in bags.keys():
        for bag in bags["neg_bags"]:
            for concept in range(num_concepts):
                proba_concepts[concept] *=  proba_concept_given_bag(env, concept, bag, positive_bag = False)

    return proba_concepts, np.argwhere(proba_concepts == np.max(proba_concepts)).flatten()

def plot_score_concepts(env, score_concepts, taboo_concepts):
    #plot the score of the concepts
    #ie. the running average of how often a concept appears as a peak
    #taboo_concepts :list of concepts we exclude for creating subgoal

    print("WARNING !!!!!!")
    print("Plot implemented only for fixed source and destination ranks (SIMPLIFIED PROBLEM)")

    destination = 0

    for people in [0,1]:

        size = len(grid_SIMPLIFIED_PB)

        grid = -1 * np.ones((size,size))

        for posId in range(env.n_positionId):

            concept = env.state2Id[posId, people, destination]

            x_concept, y_concept = env.positionId2coord[posId]

            if concept in taboo_concepts:
                grid[x_concept, y_concept] = 0
            else:
                grid[x_concept, y_concept] = score_concepts[concept]
            # grid[x_concept,y_concept] = np.log(proba_concept[concept])

            if posId == env.idxDestination2positionId[env.source_rank]:
                plt.text(y_concept, x_concept + 0.5, "source", color='r')
            elif posId == env.idxDestination2positionId[env.destination_rank]:
                plt.text(y_concept, x_concept + 0.5, "destination", color='r')

            if concept in taboo_concepts:
                plt.text(y_concept, x_concept + 0.3, "TABOO", color='r')

        ax = sns.heatmap(grid, cmap="Blues", linewidth=0.5)

        plt.title("People inside cab : " + str(people))
        plt.savefig("people_"+str(people))
        plt.show()
