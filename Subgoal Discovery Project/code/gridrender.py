from imports import *

class GUI(Canvas):
    def __init__(self, master, *args, **kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)


def draw_square_q(env, s, polygon, x, y, q, actions, dim=50):

    posId = env.id2State[s][0]
    posIdSource = env.idxDestination2positionId[env.source_rank]
    posIdDestination = env.idxDestination2positionId[env.destination_rank]
    posIdStartPoint = env.starting_location

    if posId == posIdSource :
        polygon.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='green', width=2)
    elif posId == posIdDestination :
        polygon.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='blue', width=2)
    elif posId == posIdStartPoint :
        polygon.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='yellow', width=2)
    else:
        polygon.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='white', width=2)

    font = ('Helvetica', '10', 'bold')

    for i, a in enumerate(actions):
        if a == 0:
            polygon.create_polygon([x + dim, y, x + dim / 2., y + dim / 2., x + dim, y + dim], outline='gray',
                                   fill='red', width=2)
            polygon.create_text(x + 3 * dim / 4., y + dim / 2., font=font, text="{:.3f}".format(q[i]), anchor='center')
        elif a == 1:
            polygon.create_polygon([x, y + dim, x + dim / 2., y + dim / 2., x + dim, y + dim], outline='gray',
                                   fill='green', width=2)
            polygon.create_text(x + dim / 2., y + 3 * dim / 4., font=font, text="{:.3f}".format(q[i]), anchor='n')
        elif a == 2:
            polygon.create_polygon([x, y, x + dim / 2., y + dim / 2., x, y + dim], outline='gray',
                                   fill='yellow', width=2)
            polygon.create_text(x + dim / 4., y + dim / 2., font=font, text="{:.3f}".format(q[i]), anchor='center')
        elif a == 3:
            polygon.create_polygon([x + dim, y, x + dim / 2., y + dim / 2., x, y], outline='gray',
                                   fill='purple', width=2)
            polygon.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.3f}".format(q[i]), anchor='s')


def draw_square_policy(env, s, w, x, y, pol, actions, dim=50, source=None, destination=1):

    posId = env.id2State[s][0]

    if source!=None :
        posIdSource = env.idxDestination2positionId[source]

    posIdDestination = env.idxDestination2positionId[destination]
    #posIdStartPoint = env.start_point

    if (source != None) and (posId == posIdSource) :
        w.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='green', width=2)
    elif posId == posIdDestination :
        w.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='blue', width=2)
    #elif posId == posIdStartPoint :
    #    w.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
    #                     fill='yellow', width=2)
    else:
        w.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                         fill='white', width=2)

    font = ('Helvetica', '10', 'bold')

    if (hasattr(pol, "size") and pol.size > 1) or isinstance(pol, list):
        d = pol
    else:
        d = [-1] * len(actions)
        idx = actions.index(pol)
        d[idx] = 1
        
    for j, v in enumerate(d):
        if j < len(actions):
            a = actions[j]
            if a == 0 and v > 0:
                w.create_line(x + dim / 2., y + dim / 2., x + 3*dim / 4., y + dim / 2., tags=("line",), arrow="last")
                if not np.isclose(v, 1.):
                    w.create_text(x + 3*dim / 4., y + dim / 2., font=font, text="{:.1f}".format(v), anchor='w')
            elif a == 1 and v > 0:
                w.create_line(x + dim / 2., y + dim / 2., x + dim / 2., y + 3* dim / 4., tags=("line",), arrow="last")
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + 3*dim / 4., font=font, text="{:.1f}".format(v), anchor='n')
            elif a == 2 and v >0:
                w.create_line(x + dim / 2., y + dim / 2., x+dim/4., y + dim/2., tags=("line",), arrow="last")
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 4., y + dim / 2., font=font, text="{:.1f}".format(v), anchor='e')
            elif a == 3 and v >0:
                w.create_line(x + dim / 2., y + dim / 2., x + dim / 2., y + dim / 4., tags=("line",), arrow="last")
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.1f}".format(v), anchor='s')
            elif a == 4 and v >0:
                w.create_text(x + dim / 2., y + dim / 4., font=font, text="Pick", anchor='c')
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.1f}".format(v), anchor='s')
            elif a == 5 and v >0:
                w.create_text(x + dim / 2., y + dim / 4., font=font, text="Put", anchor='c')
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.1f}".format(v), anchor='s')

def render_q(env, q):
    root = Tk()
    w = GUI(root)
    rows, cols = len(env.grid), max(map(len, env.grid))
    dim = 100
    w.config(width=cols * (dim + 12), height=rows * (dim + 12))
    
    HEXQ_test = type(env.n_states) == type([])
    if HEXQ_test:
        N = env.n_states[0]
    else :
        N = env.n_states

    for posId in range(env.n_positionId):

        s = env.state2Id[posId,0,0] # TODO : print q for each state (location, people_inside, destination_rank)

        if HEXQ_test:
            actions_list = env.state_actions[env.hierarchical_level][s]
        else :
            actions_list = env.state_actions[s]

        r, c = env.positionId2coord[posId]
        draw_square_q(env, s, w, 10 + c * (dim + 4), 10 + r * (dim + 4), dim=dim, q=q[s], actions=actions_list)
        if env.grid[r][c] == 'r' :
            w.create_line(10+(c+1)*(dim+4), 10+r*(dim+4), 10+(c+1)*(dim+4), 5+(r+1)*(dim+4), fill='red', width=10)
        
        w.pack()
    w.pack()
    root.mainloop()


def render_policy(env, d):


    for destination in env.idx_destination_ranks:

        for idx_people in range(env.num_destinations + 1):

            if idx_people!=destination:

                root = Tk()
                w = GUI(root)
                rows, cols = len(env.grid), max(map(len, env.grid))
                dim = 100
                w.config(width=cols * (dim + 12), height=rows * (dim + 12))

                if idx_people < env.num_destinations:
                    title = "people at the green point, destination at the blue point"
                    if env.SIMPLIFIED_PB:
                        source = 1
                    else:
                        source = idx_people
                else:
                    title = "people inside the cab, destination at the blue point"
                    source = None
                root.title(title)

                for posId in range(env.n_positionId):

                    s = env.state2Id[posId, idx_people, destination]

                    actions_list = env.state_actions[s]

                    r, c = env.positionId2coord[posId]
                    draw_square_policy(env, s, w, 10 + c * (dim + 4), 10 + r * (dim + 4), dim=dim, pol=d[s], actions=actions_list, source=source, destination=destination)
                    if env.grid[r][c] == 'r':
                        w.create_line(10 + (c + 1) * (dim + 4), 10 + r * (dim + 4), 10 + (c + 1) * (dim + 4),
                                      5 + (r + 1) * (dim + 4), fill='red', width=10)
                    w.pack()

                w.pack()
                root.mainloop()


