import collections
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import shutil
from sty import fg, bg, ef, rs, RgbFg
import sys



### SETTINGS ###################################################################

env = "GoToObj"

n_frames = 1

num_symbols = 8
max_len_msg = 8


### VISUALIZATION ###

sort_by = "frequency"

top = 9

show = 0

# show kth frame after message emission (-1 means average over all k)
kth = 0

# if fewer than this percentage of trajectories passed through a node,
# this node (as well as its following nodes) is not shown
threshold = 0.0


### COSMETICS ###

arrow_len = 3

lightness_scale = 1.0

def transform(v):
    return math.pow(v, 1/4)



### CHECK ######################################################################

print("Checking...", end="")
sys.stdout.flush()

assert sort_by in [  "message",
                   "frequency", "frequency_nothing", "frequency_wall","frequency_door", "frequency_key", "frequency_ball", "frequency_box",
                     "action0",   "action0_nothing",   "action0_wall",  "action0_door",   "action0_key",   "action0_ball",   "action0_box",
                     "action1",   "action1_nothing",   "action1_wall",  "action1_door",   "action1_key",   "action1_ball",   "action1_box",
                     "action2",   "action2_nothing",   "action2_wall",  "action2_door",   "action2_key",   "action2_ball",   "action2_box",
                     "action3",   "action3_nothing",   "action3_wall",  "action3_door",   "action3_key",   "action3_ball",   "action3_box",
                     "action4",   "action4_nothing",   "action4_wall",  "action4_door",   "action4_key",   "action4_ball",   "action4_box",
                     "action5",   "action5_nothing",   "action5_wall",  "action5_door",   "action5_key",   "action5_ball",   "action5_box",
                     "action6",   "action6_nothing",   "action6_wall",  "action6_door",   "action6_key",   "action6_ball",   "action6_box"]

assert -1 <= kth and kth < n_frames

assert 0.0 <= threshold and threshold <= 1.0

assert 0 <= arrow_len

assert 0.0 <= lightness_scale and lightness_scale <= 1.0

print("             done.")
sys.stdout.flush()



### INITIALIZE #################################################################

print("Loading data...", end="")
sys.stdout.flush()

# load data
data = np.load("data/%s/%d/data.npy" % (env, n_frames))

frames_actual = len(data)

if env == "GoToObj" or env == "GoToObjCustom":
    R = 6
else:
    R = 7

# dictionary message to action
d_m2a = collections.defaultdict(lambda: np.zeros((n_frames, 9, 7)))

# dictionary message to heatmap; relative (to agent) object positions
d_m2h_goal = collections.defaultdict(lambda: np.zeros((n_frames+1, 9, 2*R+1, 2*R+1)))
d_m2h      = collections.defaultdict(lambda: np.zeros((n_frames+1, 9, 2*R+1, 2*R+1, 4, 6)))

# dictionary message to heatmap; absolute object positions
d_m2h_goal_abs = collections.defaultdict(lambda: np.zeros((n_frames+1, 9, R+2, R+2)))
d_m2h_abs      = collections.defaultdict(lambda: np.zeros((n_frames+1, 9, R+2, R+2, 4, 6)))

# dictionary message to heatmap; absolute agent position
d_m2h_agt = collections.defaultdict(lambda: np.zeros((n_frames+1, 9, R+2, R+2)))

trajectories = collections.defaultdict(lambda: [0] + [[0] for i in range(63)])

print("         done.")
sys.stdout.flush()



### COLLECT ####################################################################

print("Combing through data...", end="")
sys.stdout.flush()

j = 0
for i in range(frames_actual):
    # reset if new episode
    new_episode = data[i, 3 + max_len_msg + 2]
    j *= 1 - new_episode
    
    agent_x = data[i, 0]
    agent_y = data[i, 1]

    objects_x      = []
    objects_y      = []
    objects_type   = []
    objects_color  = []
    objects_locked = []
    g = 3 + max_len_msg + 4
    while g < len(data[0]):
        object_x      = data[i, g  ]
        object_y      = data[i, g+1]
        object_type   = data[i, g+2]
        object_color  = data[i, g+3]
        object_locked = data[i, g+4]
        
        if object_x != 0:
            objects_x.append(object_x)
            objects_y.append(object_y)
            objects_type.append(object_type)
            objects_color.append(object_color)
            objects_locked.append(object_locked)
            
        g += 5
    
    if new_episode == 1:
        objects_type_ep   = objects_type
        objects_color_ep  = objects_color

    # insert obscured objects
    taken = [0]*len(objects_type)
    for b_ep in range(len(objects_type_ep)):
        found = False
        # try to find the object in the current step
        for b in range(len(objects_type)):
            if objects_type_ep[b_ep] == objects_type[b] and objects_color_ep[b_ep] == objects_color[b] and taken[b] == 0:
                taken[b] == 1
                found = True
        if not found:
            objects_x.append(agent_x)
            objects_y.append(agent_y)
            objects_type.append(objects_type_ep[b_ep])
            objects_color.append(objects_color_ep[b_ep])
            objects_locked.append(0)

    condition = 0
    for b in range(len(objects_x)):
        if objects_x[b] == agent_x and objects_y[b] == agent_y - 1:
            # agent is facing object
            if objects_type[b] == 4:
                # object is a door
                condition = 3
                if 0 < objects_locked[b]:
                    condition -= 1
                for b2 in range(len(objects_x)):
                    if objects_x[b2] == agent_x and objects_y[b2] == agent_y and objects_type[b2] == 5 and objects_color[b2] == objects_color[b]:
                        # carrying key for this door
                        condition += 2
                        break
            elif objects_type[b] == 5:
                # object is a key
                condition = 6
            elif objects_type[b] == 6:
                # object is a ball
                condition = 7
            elif objects_type[b] == 7:
                # object is a box
                condition = 8
            break
    
    if agent_y == 1 or (R == 7 and agent_y == 5 and not (2 <= condition and condition <= 5)):
        # agent is facing a wall
        condition = 1

    if j % n_frames == 0 and new_episode != 1:
        d_m2h_agt[message][n_frames][condition][agent_x][agent_y] += 1
        
        for b in range(len(objects_x)):
            x     = R + objects_x[b] - agent_x
            y     = R + objects_y[b] - agent_y
            type  = objects_type[b]
            color = objects_color[b]
            
            d_m2h[message][n_frames][condition][x][y][type-4][color] += 1
            
            d_m2h_abs[message][n_frames][condition][objects_x[b]][objects_y[b]][type-4][color] += 1
            
            if type == goal_type and color == goal_color:
                d_m2h_goal[message][n_frames][condition][x][y] += 1
                
                d_m2h_goal_abs[message][n_frames][condition][objects_x[b]][objects_y[b]] += 1

    message = tuple(data[i, 4:4+max_len_msg])
    action  = data[i, 3 + max_len_msg + 1]
    
    goal_type  = data[i, 2]
    goal_color = data[i, 3]
    
    d_m2h_agt[message][j%n_frames][condition][agent_x][agent_y] += 1
    
    for b in range(len(objects_x)):
        x     = R + objects_x[b] - agent_x
        y     = R + objects_y[b] - agent_y
        type  = objects_type[b]
        color = objects_color[b]
        
        d_m2a[message][j%n_frames][condition][action] += 1
        
        d_m2h[message][j%n_frames][condition][x][y][type-4][color] += 1
        
        d_m2h_abs[message][j%n_frames][condition][objects_x[b]][objects_y[b]][type-4][color] += 1
        
        if type == goal_type and color == goal_color:
            d_m2h_goal[message][j%n_frames][condition][x][y] += 1
            
            d_m2h_goal_abs[message][j%n_frames][condition][objects_x[b]][objects_y[b]] += 1

    # store trajectory
    if j % n_frames == 0:
        current     = trajectories[message]
        current[0] += 1
    
    if len(current) == 1:
        current += [[0] for i in range(63)]

    current = current[action+1 + condition*7]
    current[0] += 1

    j += 1

# sum total
total = np.zeros(n_frames)
for message in d_m2a:
    for i in range(n_frames):
        total[i] += d_m2a[message][i].sum()

print(" done.")
sys.stdout.flush()



### SORT #######################################################################

print("Sorting data...", end="")
sys.stdout.flush()

M = np.zeros((len(d_m2a), max_len_msg + 1))
for i, message in enumerate(d_m2a):
    M[i, :-1] = message

for i in reversed(range(max_len_msg)):
    M = M[M[:, i].argsort(kind="stable")]

def sort_by_action(action):
    if kth == -1:
        for i, message in enumerate(M[:, :-1]):
            if sort_by.endswith("_nothing"):
                s = d_m2a[tuple(message)][:, 0].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 0, action].sum() / s
            elif sort_by.endswith("_wall"):
                s = d_m2a[tuple(message)][:, 1].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 1, action].sum() / s
            elif sort_by.endswith("_door"):
                s = d_m2a[tuple(message)][:, 2:6].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 2:6, action].sum() / s
            elif sort_by.endswith("_key"):
                s = d_m2a[tuple(message)][:, 6].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 6, action].sum() / s
            elif sort_by.endswith("_ball"):
                s = d_m2a[tuple(message)][:, 7].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 7, action].sum() / s
            elif sort_by.endswith("_box"):
                s = d_m2a[tuple(message)][:, 8].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 8, action].sum() / s
            else:
                M[i, -1] = d_m2a[tuple(message)][:, :, action].sum() / d_m2a[tuple(message)].sum()
    else:
        for i, message in enumerate(M[:, :-1]):
            if sort_by.endswith("_nothing"):
                s = d_m2a[tuple(message)][kth, 0].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 0, action] / s
            elif sort_by.endswith("_wall"):
                s = d_m2a[tuple(message)][kth, 1].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 1, action] / s
            elif sort_by.endswith("_door"):
                s = d_m2a[tuple(message)][kth, 2:6].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 2:6, action].sum() / s
            elif sort_by.endswith("_key"):
                s = d_m2a[tuple(message)][kth, 6].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 6, action] / s
            elif sort_by.endswith("_ball"):
                s = d_m2a[tuple(message)][kth, 7].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 7, action] / s
            elif sort_by.endswith("_box"):
                s = d_m2a[tuple(message)][kth, 8].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 8, action] / s
            else:
                s = d_m2a[tuple(message)][kth].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, :, action].sum() / s

if sort_by != "message":
    if sort_by.startswith("frequency"):
        if kth == -1:
            for i, message in enumerate(M[:, :-1]):
                if sort_by.endswith("_nothing"):
                    M[i, -1] = d_m2a[tuple(message)][:, 0].sum()
                elif sort_by.endswith("_wall"):
                    M[i, -1] = d_m2a[tuple(message)][:, 1].sum()
                elif sort_by.endswith("_door"):
                    M[i, -1] = d_m2a[tuple(message)][:, 2:6].sum()
                elif sort_by.endswith("_key"):
                    M[i, -1] = d_m2a[tuple(message)][:, 6].sum()
                elif sort_by.endswith("_ball"):
                    M[i, -1] = d_m2a[tuple(message)][:, 7].sum()
                elif sort_by.endswith("_box"):
                    M[i, -1] = d_m2a[tuple(message)][:, 8].sum()
                else:
                    M[i, -1] = d_m2a[tuple(message)].sum()
        else:
            for i, message in enumerate(M[:, :-1]):
                if sort_by.endswith("_nothing"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 0].sum()
                elif sort_by.endswith("_wall"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 1].sum()
                elif sort_by.endswith("_door"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 2:6].sum()
                elif sort_by.endswith("_key"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 6].sum()
                elif sort_by.endswith("_ball"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 7].sum()
                elif sort_by.endswith("_box"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 8].sum()
                else:
                    M[i, -1] = d_m2a[tuple(message)][kth].sum()
    elif sort_by.startswith("action"):
        sort_by_action(int(sort_by[6]))

    M = M[M[:, -1].argsort(kind="stable")]
    M = np.flip(M, axis=0)

M = M[:, :-1]

print("         done.")
sys.stdout.flush()



### PRINT ######################################################################

def get_d_max(trajectory, d=0):
    d_max = d
    for i in range(63):
        next_trajectory = trajectory[i+1]
        if threshold*frames_actual < next_trajectory[0]:
            if 1 < len(next_trajectory):
                new_d_max = get_d_max(next_trajectory, d=d+1)
                if d_max < new_d_max:
                    d_max = new_d_max
    return d_max

got_d_max = 0
for message in trajectories:
    new_got_d_max = get_d_max(trajectories[tuple(message)])
    if got_d_max < new_got_d_max:
        got_d_max = new_got_d_max

def sort_actions(trajectory):
    X = list(range(63))
    Y = [0]*63
    for i in range(63):
        Y[i] = trajectory[i+1][0]
    return [x for _,x in reversed(sorted(zip(Y,X)))]

def next_sum(next_trajectory):
    s = 0
    for i in range(63):
        s += next_trajectory[i+1][0]
    return s

def print_trajectory(trajectory, s=None, d=0, d_max=n_frames, j=0):
    if s is None:
        s = trajectory[0]
    actions_sorted = sort_actions(trajectory)
    for i in range(63):
        if actions_sorted[i] < 7:
            condition = " "
        elif actions_sorted[i] < 14:
            condition = "w"
        elif actions_sorted[i] < 21:
            condition = "l"
        elif actions_sorted[i] < 28:
            condition = "u"
        elif actions_sorted[i] < 35:
            condition = "L"
        elif actions_sorted[i] < 42:
            condition = "U"
        elif actions_sorted[i] < 49:
            condition = "k"
        elif actions_sorted[i] < 56:
            condition = "a"
        elif actions_sorted[i] < 63:
            condition = "x"
        next_trajectory = trajectory[actions_sorted[i]+1]
        if threshold*frames_actual < next_trajectory[0]:
            q = math.floor((1 - transform(next_trajectory[0] / s)) * 24 * lightness_scale)
            if d == 0:
                if j == 0:
                    print((fg(q+232) + "   %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
                else:
                    print((fg(q+232) + "\n   %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
            else:
                if j == 0:
                    print((fg(q+232) + " " + arrow_len*"-" + "> %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
                else:
                    print((fg(q+232) + "\n" + (5+(d-1)*(5+arrow_len))*" " + " " + arrow_len*"-" + "> %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
            if 1 < len(next_trajectory):
                n_s = next_sum(next_trajectory)
                if next_trajectory[0] == n_s:
                    print_trajectory(next_trajectory, s, d=d+1, d_max=d_max)
                else:
                    print((fg(q+232) + (d_max-d)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (next_trajectory[0] - n_s, next_trajectory[0], (next_trajectory[0] - n_s)/next_trajectory[0]*100), end="")
                    print_trajectory(next_trajectory, s, d=d+1, d_max=d_max, j=1)
            else:
                print((fg(q+232) + (d_max-d)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (next_trajectory[0], next_trajectory[0], 100), end="")
            j += 1
    if d != 0 and j == 0:
        q = math.floor((1 - transform(trajectory[0] / s)) * 24 * lightness_scale)
        n_s = next_sum(trajectory)
        print((fg(q+232) + (d_max-d+1)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (trajectory[0] - n_s, trajectory[0], (trajectory[0] - n_s)/trajectory[0]*100), end="")

def action_probs(cond, m, k):
    print("%11s   " % cond, end="")
    if cond == "all":
        if k == -1:
            actions = d_m2a[tuple(m)][:, :].sum(0).sum(0)
        else:
            actions = d_m2a[tuple(m)][k].sum(0)
    else:
        if cond == "nothing":
            c = 0
        elif cond == "wall":
            c = 1
        elif cond == "door_l":
            c = 2
        elif cond == "door_u":
            c = 3
        elif cond == "door_l+k":
            c = 4
        elif cond == "door_u+k":
            c = 5
        elif cond == "key":
            c = 6
        elif cond == "ball":
            c = 7
        elif cond == "box":
            c = 8
        if k == -1:
            actions = d_m2a[tuple(m)][:, c].sum(0)
        else:
            actions = d_m2a[tuple(m)][k, c]
    sum = actions.sum()
    if 0 < sum:
        actions /= sum
    for action in actions:
        print("%03.2f   " % action, end="")
    print("%7d" % sum, end="")
    if k == -1:
        print("   %7.4f" % (sum/total.sum()*100))
    else:
        print("   %7.4f" % (sum/total[k].sum()*100))

for i, m in enumerate(M[:top]):
    print("\n\n")
    print("%2d " % i, end="")
    for s in m:
        print(chr(97+int(s+0.5)), end="")
    print()
    print("                                   actions")
    print("                 0      1      2      3      4      5      6")
    print("        cdn    lft    rgt    fwd    pkp    drp    tgl    dne       frq     % tot")
    print("   -----------------------------------------------------------------------------")

    action_probs("all",      m, kth)
    action_probs("nothing",  m, kth)
    action_probs("wall",     m, kth)
    action_probs("door_l",   m, kth)
    action_probs("door_u",   m, kth)
    action_probs("door_l+k", m, kth)
    action_probs("door_u+k", m, kth)
    action_probs("key",      m, kth)
    action_probs("ball",     m, kth)
    action_probs("box",      m, kth)

    print("\n" + (5 + got_d_max*(5+arrow_len))*" " + "      stop / pass        % stop")
    print_trajectory(trajectories[tuple(m)], d_max=got_d_max)
    print()



### PLOT #######################################################################

# Used to map colors to integers
COLOR_TO_IDX = {
    "red"   : 0,
    "green" : 1,
    "blue"  : 2,
    "purple": 3,
    "yellow": 4,
    "grey"  : 5
}

# Map of object type to integers
OBJECT_TO_IDX = {
#    "unseen"        : 0,
#    "empty"         : 1,
#    "wall"          : 2,
#    "floor"         : 3,
    "door"          : 0,
    "key"           : 1,
    "ball"          : 2,
    "box"           : 3,
#    "goal"          : 8,
#    "lava"          : 9
}

#for message in d_m2h:
    # object distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, :, :].sum(-1).sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, :, :].sum(-1).sum(-1)

    # ball distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, OBJECT_TO_IDX["ball"], :].sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, OBJECT_TO_IDX["ball"], :].sum(-1)

    # blue object distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, :, COLOR_TO_IDX["blue"]].sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, :, COLOR_TO_IDX["blue"]].sum(-1)

    # blue ball distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, OBJECT_TO_IDX["ball"], COLOR_TO_IDX["blue"]]
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, OBJECT_TO_IDX["ball"], COLOR_TO_IDX["blue"]]

    # key distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, OBJECT_TO_IDX["key"], :].sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, OBJECT_TO_IDX["key"], :].sum(-1)

# goal distribution
d_m2h     = d_m2h_goal
d_m2h_abs = d_m2h_goal_abs

def normalize(h):
    h_new = h.copy()
    s = h_new.sum()
    if 0 < s:
        h_new /= s
    return h_new.transpose()

def addagent(h, q=32):
    assert q == 32
    
    cell = h[R*q:(R+1)*q, R*q:(R+1)*q, :]
    
    cell[ 2   , 15   , :] =         np.array([1, 0, 0, 1])
    cell[ 3: 4, 15:17, :] = np.tile(np.array([1, 0, 0, 1]), [1,  2, 1])
    cell[ 4: 5, 14:17, :] = np.tile(np.array([1, 0, 0, 1]), [1,  3, 1])
    cell[ 5: 7, 14:18, :] = np.tile(np.array([1, 0, 0, 1]), [1,  4, 1])
    cell[ 7: 9, 13:19, :] = np.tile(np.array([1, 0, 0, 1]), [1,  6, 1])
    cell[ 9:10, 12:19, :] = np.tile(np.array([1, 0, 0, 1]), [1,  7, 1])
    cell[10:12, 12:20, :] = np.tile(np.array([1, 0, 0, 1]), [1,  8, 1])
    cell[12:14, 11:21, :] = np.tile(np.array([1, 0, 0, 1]), [1, 10, 1])
    cell[14:15, 10:21, :] = np.tile(np.array([1, 0, 0, 1]), [1, 11, 1])
    cell[15:16, 10:22, :] = np.tile(np.array([1, 0, 0, 1]), [1, 12, 1])
    cell[16:17,  9:22, :] = np.tile(np.array([1, 0, 0, 1]), [1, 13, 1])
    cell[17:19,  9:23, :] = np.tile(np.array([1, 0, 0, 1]), [1, 14, 1])
    cell[19:21,  8:24, :] = np.tile(np.array([1, 0, 0, 1]), [1, 16, 1])
    cell[21:22,  7:24, :] = np.tile(np.array([1, 0, 0, 1]), [1, 17, 1])
    cell[22:24,  7:25, :] = np.tile(np.array([1, 0, 0, 1]), [1, 18, 1])
    cell[24:27,  6:26, :] = np.tile(np.array([1, 0, 0, 1]), [1, 20, 1])
    
    h[R*q:(R+1)*q, R*q:(R+1)*q, :] = cell
    
    # red square
    #h[R*q:(R+1)*q, R*q:(R+1)*q, :] = np.tile(np.array([1, 0, 0, 1]), [q, q, 1])
    
    return h

def expand(h, q=32):
    N = h.shape[0]
    M = h.shape[1]
    Z = h.shape[2]
    h_new = np.zeros((N*q, M*q, Z))
    for i in range(N):
        for j in range(M):
            h_new[i*q:(i+1)*q, j*q:(j+1)*q, :] = np.tile(h[i, j, :], [q, q, 1])
    return h_new

def imshow2(ax, h, t="", add_agent=False):
    ax.set_title(t)
    norm = plt.Normalize(0.0, h.max())
    rgba = expand(cmap(norm(h)))
    if add_agent:
        rgba = addagent(rgba)
    im = ax.imshow(h, cmap=cmap, vmin=0.0, vmax=h.max())
    ax.imshow(rgba)
    fig.colorbar(im, ax=ax, orientation="vertical")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def imsave2(fname, h, cmap, add_agent=False):
    norm = plt.Normalize(0.0, h.max())
    rgba = expand(cmap(norm(h)))
    if add_agent:
        rgba = addagent(rgba)
    plt.imsave(fname, rgba, format="png")

def imssave2(d_m2, str):
    print("Saving %s files..." % str, end="")
    sys.stdout.flush()
    
    if str == "rel" or str == "abs":
        cmap = plt.cm.cividis
    elif str == "agt":
        cmap = plt.cm.inferno
    
    if str == "rel":
        add_agent = True
    else:
        add_agent = False

    for m in M[:top]:
        t = ""
        for s in m:
            t += chr(97+int(s))
        
        if not os.path.exists("renders/%s" % t):
            os.mkdir("renders/%s" % t)
        if not os.path.exists("renders/%s/%s" % (t, str)):
            os.mkdir("renders/%s/%s" % (t, str))
        if not os.path.exists("renders/%s/%s/all" % (t, str)):
            os.mkdir("renders/%s/%s/all" % (t, str))
        if not os.path.exists("renders/%s/%s/nothing" % (t, str)):
            os.mkdir("renders/%s/%s/nothing" % (t, str))
        if not os.path.exists("renders/%s/%s/wall" % (t, str)):
            os.mkdir("renders/%s/%s/wall" % (t, str))
        if not os.path.exists("renders/%s/%s/door" % (t, str)):
            os.mkdir("renders/%s/%s/door" % (t, str))
        if not os.path.exists("renders/%s/%s/door/locked" % (t, str)):
            os.mkdir("renders/%s/%s/door/locked" % (t, str))
        if not os.path.exists("renders/%s/%s/door/locked/key" % (t, str)):
            os.mkdir("renders/%s/%s/door/locked/key" % (t, str))
        if not os.path.exists("renders/%s/%s/door/unlocked" % (t, str)):
            os.mkdir("renders/%s/%s/door/unlocked" % (t, str))
        if not os.path.exists("renders/%s/%s/door/unlocked/key" % (t, str)):
            os.mkdir("renders/%s/%s/door/unlocked/key" % (t, str))
        if not os.path.exists("renders/%s/%s/key" % (t, str)):
            os.mkdir("renders/%s/%s/key" % (t, str))
        if not os.path.exists("renders/%s/%s/ball" % (t, str)):
            os.mkdir("renders/%s/%s/ball" % (t, str))
        if not os.path.exists("renders/%s/%s/box" % (t, str)):
            os.mkdir("renders/%s/%s/box" % (t, str))

        h_a_agt   = normalize(d_m2[tuple(m)][:-1].sum(0).sum(0))
        h_0_agt   = normalize(d_m2[tuple(m)][:-1, 0].sum(0))
        h_1_agt   = normalize(d_m2[tuple(m)][:-1, 1].sum(0))
        h_2_6_agt = normalize(d_m2[tuple(m)][:-1, 2:6].sum(0).sum(0))
        h_2_agt   = normalize(d_m2[tuple(m)][:-1, 2].sum(0))
        h_3_agt   = normalize(d_m2[tuple(m)][:-1, 3].sum(0))
        h_4_agt   = normalize(d_m2[tuple(m)][:-1, 4].sum(0))
        h_5_agt   = normalize(d_m2[tuple(m)][:-1, 5].sum(0))
        h_6_agt   = normalize(d_m2[tuple(m)][:-1, 6].sum(0))
        h_7_agt   = normalize(d_m2[tuple(m)][:-1, 7].sum(0))
        h_8_agt   = normalize(d_m2[tuple(m)][:-1, 8].sum(0))
        
        imsave2("renders/%s/%s/all/k_avg.png"               % (t, str), h_a_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/nothing/k_avg.png"           % (t, str), h_0_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/wall/k_avg.png"              % (t, str), h_1_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/door/k_avg.png"              % (t, str), h_2_6_agt, cmap, add_agent)
        imsave2("renders/%s/%s/door/locked/k_avg.png"       % (t, str), h_2_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/door/unlocked/k_avg.png"     % (t, str), h_3_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/door/locked/key/k_avg.png"   % (t, str), h_4_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/door/unlocked/key/k_avg.png" % (t, str), h_5_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/key/k_avg.png"               % (t, str), h_6_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/ball/k_avg.png"              % (t, str), h_7_agt,   cmap, add_agent)
        imsave2("renders/%s/%s/box/k_avg.png"               % (t, str), h_8_agt,   cmap, add_agent)
        
        for j in range(n_frames+1):
            h_a_agt   = normalize(d_m2[tuple(m)][j].sum(0))
            h_0_agt   = normalize(d_m2[tuple(m)][j, 0])
            h_1_agt   = normalize(d_m2[tuple(m)][j, 1])
            h_2_6_agt = normalize(d_m2[tuple(m)][j, 2:6].sum(0))
            h_2_agt   = normalize(d_m2[tuple(m)][j, 2])
            h_3_agt   = normalize(d_m2[tuple(m)][j, 3])
            h_4_agt   = normalize(d_m2[tuple(m)][j, 4])
            h_5_agt   = normalize(d_m2[tuple(m)][j, 5])
            h_6_agt   = normalize(d_m2[tuple(m)][j, 6])
            h_7_agt   = normalize(d_m2[tuple(m)][j, 7])
            h_8_agt   = normalize(d_m2[tuple(m)][j, 8])
            
            imsave2("renders/%s/%s/all/k%d.png"               % (t, str, j), h_a_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/nothing/k%d.png"           % (t, str, j), h_0_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/wall/k%d.png"              % (t, str, j), h_1_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/door/k%d.png"              % (t, str, j), h_2_6_agt, cmap, add_agent)
            imsave2("renders/%s/%s/door/locked/k%d.png"       % (t, str, j), h_2_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/door/unlocked/k%d.png"     % (t, str, j), h_3_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/door/locked/key/k%d.png"   % (t, str, j), h_4_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/door/unlocked/key/k%d.png" % (t, str, j), h_5_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/key/k%d.png"               % (t, str, j), h_6_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/ball/k%d.png"              % (t, str, j), h_7_agt,   cmap, add_agent)
            imsave2("renders/%s/%s/box/k%d.png"               % (t, str, j), h_8_agt,   cmap, add_agent)
            
    print("        done.")
    sys.stdout.flush()

m = M[show]

t = ""
for s in m:
    t += chr(97+int(s))

### FIGURE 1 ###

cmap = plt.cm.cividis

fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a = normalize(d_m2h[tuple(m)][:-1].sum(0).sum(0))
    h_0 = normalize(d_m2h[tuple(m)][:-1, 0].sum(0))
    h_1 = normalize(d_m2h[tuple(m)][:-1, 1].sum(0))
    #h_2 = normalize(d_m2h[tuple(m)][:-1, 2:6].sum(0).sum(0))
    h_2 = normalize(d_m2h[tuple(m)][:-1, 2].sum(0))
    h_3 = normalize(d_m2h[tuple(m)][:-1, 3].sum(0))
    h_4 = normalize(d_m2h[tuple(m)][:-1, 4].sum(0))
    h_5 = normalize(d_m2h[tuple(m)][:-1, 5].sum(0))
    h_6 = normalize(d_m2h[tuple(m)][:-1, 6].sum(0))
    h_7 = normalize(d_m2h[tuple(m)][:-1, 7].sum(0))
    h_8 = normalize(d_m2h[tuple(m)][:-1, 8].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a = normalize(d_m2h[tuple(m)][kth].sum(0))
    h_0 = normalize(d_m2h[tuple(m)][kth, 0])
    h_1 = normalize(d_m2h[tuple(m)][kth, 1])
    #h_2 = normalize(d_m2h[tuple(m)][kth, 2:6].sum(0))
    h_2 = normalize(d_m2h[tuple(m)][kth, 2])
    h_3 = normalize(d_m2h[tuple(m)][kth, 3])
    h_4 = normalize(d_m2h[tuple(m)][kth, 4])
    h_5 = normalize(d_m2h[tuple(m)][kth, 5])
    h_6 = normalize(d_m2h[tuple(m)][kth, 6])
    h_7 = normalize(d_m2h[tuple(m)][kth, 7])
    h_8 = normalize(d_m2h[tuple(m)][kth, 8])

imshow2(ax0, h_a, r"$P(g | m, k)$",                                        add_agent=True)
imshow2(ax1, h_0, r"$P(g | m, k, \mathrm{nothing})$",                      add_agent=True)
imshow2(ax2, h_1, r"$P(g | m, k, \mathrm{wall})$",                         add_agent=True)
#imshow2(ax3, h_2, r"$P(g | m, k, \mathrm{door})$",                         add_agent=True)
imshow2(ax3, h_2, r"$P(g | m, k, \mathrm{locked\,door})$",                 add_agent=True)
imshow2(ax4, h_3, r"$P(g | m, k, \mathrm{unlocked\,door})$",               add_agent=True)
imshow2(ax5, h_4, r"$P(g | m, k, \mathrm{locked\,door}, \mathrm{key})$",   add_agent=True)
imshow2(ax6, h_5, r"$P(g | m, k, \mathrm{unlocked\,door}, \mathrm{key})$", add_agent=True)
imshow2(ax7, h_6, r"$P(g | m, k, \mathrm{key})$",                          add_agent=True)
imshow2(ax8, h_7, r"$P(g | m, k, \mathrm{ball})$",                         add_agent=True)
imshow2(ax9, h_8, r"$P(g | m, k, \mathrm{box})$",                          add_agent=True)

### FIGURE 2 ###

cmap = plt.cm.cividis

fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a_abs = normalize(d_m2h_abs[tuple(m)][:-1].sum(0).sum(0))
    h_0_abs = normalize(d_m2h_abs[tuple(m)][:-1, 0].sum(0))
    h_1_abs = normalize(d_m2h_abs[tuple(m)][:-1, 1].sum(0))
    #h_2_abs = normalize(d_m2h_abs[tuple(m)][:-1, 2:6].sum(0).sum(0))
    h_2_abs = normalize(d_m2h_abs[tuple(m)][:-1, 2].sum(0))
    h_3_abs = normalize(d_m2h_abs[tuple(m)][:-1, 3].sum(0))
    h_4_abs = normalize(d_m2h_abs[tuple(m)][:-1, 4].sum(0))
    h_5_abs = normalize(d_m2h_abs[tuple(m)][:-1, 5].sum(0))
    h_6_abs = normalize(d_m2h_abs[tuple(m)][:-1, 6].sum(0))
    h_7_abs = normalize(d_m2h_abs[tuple(m)][:-1, 7].sum(0))
    h_8_abs = normalize(d_m2h_abs[tuple(m)][:-1, 8].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a_abs = normalize(d_m2h_abs[tuple(m)][kth].sum(0))
    h_0_abs = normalize(d_m2h_abs[tuple(m)][kth, 0])
    h_1_abs = normalize(d_m2h_abs[tuple(m)][kth, 1])
    #h_2_abs = normalize(d_m2h_abs[tuple(m)][kth, 2:6].sum(0))
    h_2_abs = normalize(d_m2h_abs[tuple(m)][kth, 2])
    h_3_abs = normalize(d_m2h_abs[tuple(m)][kth, 3])
    h_4_abs = normalize(d_m2h_abs[tuple(m)][kth, 4])
    h_5_abs = normalize(d_m2h_abs[tuple(m)][kth, 5])
    h_6_abs = normalize(d_m2h_abs[tuple(m)][kth, 6])
    h_7_abs = normalize(d_m2h_abs[tuple(m)][kth, 7])
    h_8_abs = normalize(d_m2h_abs[tuple(m)][kth, 8])

imshow2(ax0, h_a_abs, r"$P(g | m, k)$")
imshow2(ax1, h_0_abs, r"$P(g | m, k, \mathrm{nothing})$")
imshow2(ax2, h_1_abs, r"$P(g | m, k, \mathrm{wall})$")
#imshow2(ax3, h_2_abs, r"$P(g | m, k, \mathrm{door})$")
imshow2(ax3, h_2_abs, r"$P(g | m, k, \mathrm{locked\,door})$")
imshow2(ax4, h_3_abs, r"$P(g | m, k, \mathrm{unlocked\,door})$")
imshow2(ax5, h_4_abs, r"$P(g | m, k, \mathrm{locked\,door}, \mathrm{key})$")
imshow2(ax6, h_5_abs, r"$P(g | m, k, \mathrm{unlocked\,door}, \mathrm{key})$")
imshow2(ax7, h_6_abs, r"$P(g | m, k, \mathrm{key})$")
imshow2(ax8, h_7_abs, r"$P(g | m, k, \mathrm{ball})$")
imshow2(ax9, h_8_abs, r"$P(g | m, k, \mathrm{box})$")

### FIGURE 3 ###

cmap = plt.cm.inferno

fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a_agt = normalize(d_m2h_agt[tuple(m)][:-1].sum(0).sum(0))
    h_0_agt = normalize(d_m2h_agt[tuple(m)][:-1, 0].sum(0))
    h_1_agt = normalize(d_m2h_agt[tuple(m)][:-1, 1].sum(0))
    #h_2_agt = normalize(d_m2h_agt[tuple(m)][:-1, 2:6].sum(0).sum(0))
    h_2_agt = normalize(d_m2h_agt[tuple(m)][:-1, 2].sum(0))
    h_3_agt = normalize(d_m2h_agt[tuple(m)][:-1, 3].sum(0))
    h_4_agt = normalize(d_m2h_agt[tuple(m)][:-1, 4].sum(0))
    h_5_agt = normalize(d_m2h_agt[tuple(m)][:-1, 5].sum(0))
    h_6_agt = normalize(d_m2h_agt[tuple(m)][:-1, 6].sum(0))
    h_7_agt = normalize(d_m2h_agt[tuple(m)][:-1, 7].sum(0))
    h_8_agt = normalize(d_m2h_agt[tuple(m)][:-1, 8].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a_agt = normalize(d_m2h_agt[tuple(m)][kth].sum(0))
    h_0_agt = normalize(d_m2h_agt[tuple(m)][kth, 0])
    h_1_agt = normalize(d_m2h_agt[tuple(m)][kth, 1])
    #h_2_agt = normalize(d_m2h_agt[tuple(m)][kth, 2:6].sum(0))
    h_2_agt = normalize(d_m2h_agt[tuple(m)][kth, 2])
    h_3_agt = normalize(d_m2h_agt[tuple(m)][kth, 3])
    h_4_agt = normalize(d_m2h_agt[tuple(m)][kth, 4])
    h_5_agt = normalize(d_m2h_agt[tuple(m)][kth, 5])
    h_6_agt = normalize(d_m2h_agt[tuple(m)][kth, 6])
    h_7_agt = normalize(d_m2h_agt[tuple(m)][kth, 7])
    h_8_agt = normalize(d_m2h_agt[tuple(m)][kth, 8])

imshow2(ax0, h_a_agt, r"$P(a | m, k)$")
imshow2(ax1, h_0_agt, r"$P(a | m, k, \mathrm{nothing})$")
imshow2(ax2, h_1_agt, r"$P(a | m, k, \mathrm{wall})$")
#imshow2(ax3, h_2_agt, r"$P(a | m, k, \mathrm{door})$")
imshow2(ax3, h_2_agt, r"$P(g | m, k, \mathrm{locked\,door})$")
imshow2(ax4, h_3_agt, r"$P(g | m, k, \mathrm{unlocked\,door})$")
imshow2(ax5, h_4_agt, r"$P(g | m, k, \mathrm{locked\,door}, \mathrm{key})$")
imshow2(ax6, h_5_agt, r"$P(g | m, k, \mathrm{unlocked\,door}, \mathrm{key})$")
imshow2(ax7, h_6_agt, r"$P(g | m, k, \mathrm{key})$")
imshow2(ax8, h_7_agt, r"$P(g | m, k, \mathrm{ball})$")
imshow2(ax9, h_8_agt, r"$P(g | m, k, \mathrm{box})$")

plt.pause(0.05)

### SAVE ###

print()

if not os.path.exists("renders"):
    os.mkdir("renders")
else:
    print("Deleting existing files...", end="")
    sys.stdout.flush()
    for file in os.listdir("renders"):
        file_path = os.path.join("renders", file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print(" done.")
    sys.stdout.flush()

imssave2(d_m2h,     "rel")
imssave2(d_m2h_abs, "abs")
imssave2(d_m2h_agt, "agt")

plt.show()
