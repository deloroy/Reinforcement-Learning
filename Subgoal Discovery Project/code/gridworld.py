import numpy as np
import numbers
import gridrender as gui
from tkinter import Tk
import tkinter.font as tkFont
import random


class GridWorld():
    def __init__(self, SIMPLIFIED_PB, gamma=0.95, grid=None, render=False):
        #if simplified_PB is True, we work on the Simplified Taxi Domain
        #if False, we work on the Original Taxi Domain
        #description in the report

        self.SIMPLIFIED_PB = SIMPLIFIED_PB

        self.grid = grid

        self.action_names = np.array(['right', 'down', 'left', 'up', 'pickup', 'putdown'])
        self.n_actions = 6

        if self.SIMPLIFIED_PB:#2 fixed ranks (1 for source, 1 for destination)
            self.idx_ranks = [0, 1] #source or destination
            self.idx_destination_ranks = [0] #destination
        else:
            #4 ranks (can be source or destination), taxi knows location passengers
            self.idx_ranks = [0, 1, 2, 3]  #source or destination
            self.idx_destination_ranks = [0, 1, 2, 3] #destination

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to a position id called positionId

        self.coord2positionId = np.empty_like(self.grid, dtype=np.int)
        self.positionId2coord = []

        n_positionId = 0
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'x':
                    self.coord2positionId[i, j] = n_positionId
                    n_positionId += 1
                    self.positionId2coord.append([i, j])
                else:
                    self.coord2positionId[i, j] = -1

        self.n_positionId = n_positionId

        # Create a map to translate the index of a destination to its positionId

        if self.SIMPLIFIED_PB:
            self.idxDestination2positionId = [0, 47]
        else:
            self.idxDestination2positionId = [0, 4, 20, 23]
        self.positionId2idxDestination = {pos:idx for (idx,pos) in enumerate(self.idxDestination2positionId)}

        # Create a map to translate a state (positionId,idx_people,idx_destination) to scalar index called Id
        # idx_people = num_destinations <-> people inside the cab

        num_destinations = len(self.idx_destination_ranks)
        self.num_destinations  = num_destinations

        self.n_states = n_positionId*(num_destinations+1)*num_destinations #people inside or not, destination
        self.id2State = [None] * self.n_states
        self.state2Id = -1 * np.ones((n_positionId, num_destinations+1, num_destinations),dtype=np.int)

        for positionId in range(self.n_positionId):
            for idx_people in range(num_destinations+1):
                for idx_destination in range(num_destinations):
                    id = ((num_destinations+1)*idx_destination + idx_people)*self.n_positionId + positionId
                    self.id2State[id] = [positionId,idx_people,idx_destination]
                    self.state2Id[positionId,idx_people,idx_destination] = id

        # compute the actions available in each state
        self.compute_available_actions()
        self.gamma = gamma
        self.proba_succ = 1 #0.9
        self.render = render
        self.initial_prob=None

    def reset(self):
        """
        Choose random source and destination rank
        Returns:
            An initial state randomly drawn from a uniform distribution
        """

        # initial state of the cab
        self.starting_location = random.choice(list(set(list(range(24)))-set(self.idxDestination2positionId)))

        # initial
        if self.SIMPLIFIED_PB:
            self.source_rank = 1
            self.destination_rank = 0
            self.idx_people = 0  # if SIMPLIFIED_PB, idx_people means people_inside ? (0 False, 1 True)
        else:
            idx_ranks = self.idx_ranks.copy()
            self.source_rank = random.choice(idx_ranks)
            idx_ranks.remove(self.source_rank)
            self.destination_rank = random.choice(idx_ranks)
            self.idx_people = self.source_rank  #  idx_people means idx_rank where people are located if < 4, and people inside the cab if = 4

        idx_people =  self.idx_people # we know people are at the source

        self.start_state = self.state2Id[self.starting_location, idx_people,self.destination_rank]

        self.reset_window = False
        if self.render:
            self.reset_window = True

        return self.start_state

    def step(self, id_state, action):
        """
        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        state = self.id2State[id_state]
        positionId = state[0]
        r,c = self.positionId2coord[positionId]
        idx_people = state[1]
        idx_destination = state[2]

        positionIdSource = self.idxDestination2positionId[self.source_rank]
        positionIdDestination = self.idxDestination2positionId[self.destination_rank]

        assert (action in self.state_actions[id_state])
        if isinstance(self.grid[r][c], numbers.Number):
            return id_state, 0, True
        else:
            if np.random.rand(1) > self.proba_succ :
                action = random.choice(self.state_actions[id_state])
                
            if action == 0:
                c = min(self.n_cols - 1, c + 1)
                next_state = self.state2Id[self.coord2positionId[r,c], idx_people, idx_destination]
            elif action == 1:
                r = min(self.n_rows - 1, r + 1)
                next_state = self.state2Id[self.coord2positionId[r,c], idx_people, idx_destination]
            elif action == 2:
                c = max(0, c - 1)
                next_state = self.state2Id[self.coord2positionId[r,c], idx_people, idx_destination]
            elif action == 3:
                r = max(0, r - 1) 
                next_state = self.state2Id[self.coord2positionId[r,c], idx_people, idx_destination]

            elif (action==4 and positionId == positionIdSource):
                next_state = self.state2Id[self.coord2positionId[r,c], self.num_destinations, idx_destination]

            elif (action==5 and positionId == positionIdDestination):
                #necessarily, self.people_inside==True
                #we forbid to put down when not at the destination
                next_state = self.state2Id[self.coord2positionId[r,c], idx_destination, idx_destination]

            else: #pick up but not at the source, or put down but not at the destination
                next_state = id_state

            absorb = False
            if (action==4 and positionId != positionIdSource): #wrong pick up
                #necessarily, self.people_inside==False
                reward = -10
            elif (action==4 and positionId == positionIdSource):#good pick up
                #necessarily, self.people_inside==False
                self.hierarchical_level = 1 #exit
                self.idx_people = self.num_destinations
                reward = 0
            elif (action==5 and positionId != positionIdDestination): #wrong put down
                #necessarily, self.people_inside==True
                #the passenger is not put down if not at the destination
                reward = -10
            elif (action==5 and positionId == positionIdDestination): #good put down
                #necessarily, self.people_inside==True
                #the passenger is not put down if not at the destination
                self.hierarchical_level = 2 #exit
                self.idx_people = idx_destination
                reward = 20
                absorb = True 
                
            else: #navigation   
                reward = -1           
                    
        if self.render:
            self.show(id_state, action, next_state, reward)

        return next_state, reward, absorb

    def show(self, state, action, next_state, reward):

        posIdSource = self.idxDestination2positionId[self.source_rank]
        posIdDestination = self.idxDestination2positionId[self.destination_rank]
        posIdStartPoint = self.starting_location

        dim = 100
        rows, cols = len(self.grid) + 0.5, max(map(len, self.grid))

        if self.reset_window or not hasattr(self, 'window'):
            if self.reset_window:
                self.reset_window=False
            root = Tk()
            self.window = gui.GUI(root)
            self.window.config(width=cols * (dim + 12), height=rows * (dim + 12))
            my_font = tkFont.Font(family="Arial", size=10, weight="bold")

            for posId in range(self.n_positionId):

                r, c = self.positionId2coord[posId]
                x, y = 10+c*(dim + 4), 10+r*(dim + 4)
                
                if self.grid[r][c] == 'r' :
                    self.window.create_line(10+(c+1)*(dim+4), 10+r*(dim+4), 10+(c+1)*(dim+4), 5+(r+1)*(dim+4), fill='red', width=10)

                if posId == posIdSource:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black', fill='green', width=2)
                elif posId == posIdDestination:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black', fill='blue', width=2)
                elif posId == posIdStartPoint:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black', fill='yellow', width=2)
                else:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black', fill='white', width=2)

            self.window.pack()

        my_font = tkFont.Font(family="Arial", size=10, weight="bold")

        posId = self.id2State[state][0]
        r0, c0 = self.positionId2coord[posId]
        r0, c0 = 10 + c0 * (dim + 4), 10 + r0 * (dim + 4)
        x0, y0 = r0 + dim / 2., c0 + dim / 2.

        posId = self.id2State[next_state][0]
        r1, c1 = self.positionId2coord[posId]
        r1, c1 = 10 + c1 * (dim + 4), 10 + r1 * (dim + 4)
        x1, y1 = r1 + dim / 2., c1 + dim / 2.

        if hasattr(self, 'oval2'):
            # self.window.delete(self.line1)
            # self.window.delete(self.oval1)
            self.window.delete(self.oval2)
            self.window.delete(self.text1)
            self.window.delete(self.text2)
            self.window.delete(self.text3)

        # self.line1 = self.window.create_arc(x0, y0, x1, y1, dash=(3,5))
        # self.oval1 = self.window.create_oval(x0 - dim / 20., y0 - dim / 20., x0 + dim / 20., y0 + dim / 20., dash=(3,5))
        self.oval2 = self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5., fill='red')
        self.text1 = self.window.create_text(dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="r= {:.1f}".format(reward), anchor='center')
        self.text2 = self.window.create_text(2 * dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="action: {}".format(self.action_names[action]), anchor='center')
        self.text3 = self.window.create_text(4 * dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="People in the cab: {}".format(self.idx_people==self.num_destinations), anchor='center')

        self.window.update()

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up, 4=pick, 5=put
        self.state_actions = []

        for id_state in range(self.n_states):

            state = self.id2State[id_state]

            posId = state[0]
            idx_people = state[1]

            i, j = self.positionId2coord[posId]
            actions = [0, 1, 2, 3, 4, 5]

            if i == 0:
                actions.remove(3)
            elif i == self.n_rows - 1:
                actions.remove(1)
            if j == 0 or self.grid[i][j] == 'l':
                actions.remove(2)
            if j == self.n_cols-1 or self.grid[i][j] == 'r':
                actions.remove(0)

            if idx_people==self.num_destinations: #assuming we can't load more than one person
                actions.remove(4)

            else: #we can't put down nobody
                actions.remove(5)

            self.state_actions.append(actions)

    def describe_state(self, id_state):
        descr_state = self.id2State[id_state]
        print("position : ", self.positionId2coord[descr_state[0]])
        tmp = "passengers in my cab : " + str(descr_state[1] == self.num_destinations)
        print(tmp)
        print("destination : ", self.idxDestination2positionId[descr_state[2]])


###############
###############

#DEFINING THE TWO TAXI ENVIRONMENT


grid_SIMPLIFIED_PB = [ ['']*7] *7
GridWorld_SIMPLIFIED_PB = GridWorld(True, gamma=0.95, grid=grid_SIMPLIFIED_PB)  #source and destination ranks change


grid_ORIGINAL_PB = [
    ['', 'r','l', '', ''],
    ['', 'r','l', '', ''],
    ['', '','', '', ''],
    ['r', 'l','r', 'l', ''],
    ['r', 'l','r', 'l', '']
]
GridWorld_ORIGINAL_PB = GridWorld(False, gamma=0.95, grid=grid_ORIGINAL_PB)  #source and destination ranks change

