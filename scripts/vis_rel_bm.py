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

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}\usepackage{times}\usepackage{latexsym}"]



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
threshold = 0.001


### COSMETICS ###

arrow_len = 3

lightness_scale = 1.0

def transform(v):
    return math.pow(v, 1/4)



### CHECK ######################################################################

print("Checking...", end="")
sys.stdout.flush()

assert sort_by in [  "message",
                   "frequency", "frequency_nothing", "frequency_wall", "frequency_door", "frequency_key", "frequency_ball", "frequency_box",
                     "action0",   "action0_nothing",   "action0_wall",   "action0_door",   "action0_key",   "action0_ball",   "action0_box",
                     "action1",   "action1_nothing",   "action1_wall",   "action1_door",   "action1_key",   "action1_ball",   "action1_box",
                     "action2",   "action2_nothing",   "action2_wall",   "action2_door",   "action2_key",   "action2_ball",   "action2_box",
                     "action3",   "action3_nothing",   "action3_wall",   "action3_door",   "action3_key",   "action3_ball",   "action3_box",
                     "action4",   "action4_nothing",   "action4_wall",   "action4_door",   "action4_key",   "action4_ball",   "action4_box",
                     "action5",   "action5_nothing",   "action5_wall",   "action5_door",   "action5_key",   "action5_ball",   "action5_box",
                     "action6",   "action6_nothing",   "action6_wall",   "action6_door",   "action6_key",   "action6_ball",   "action6_box"]

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

#
p2m_goal_abs = [[collections.defaultdict(lambda: 0) for j in range(R+2)] for i in range(R+2)]

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
                if 1 < objects_locked[b]:
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

    d_m2a[message][j%n_frames][condition][action] += 1

    d_m2h_agt[message][j%n_frames][condition][agent_x][agent_y] += 1
    
    for b in range(len(objects_x)):
        x     = R + objects_x[b] - agent_x
        y     = R + objects_y[b] - agent_y
        type  = objects_type[b]
        color = objects_color[b]
        
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

    # store *2m
    if j % n_frames == 0:
        for b in range(len(objects_x)):
            type  = objects_type[b]
            color = objects_color[b]
            if type == goal_type and color == goal_color:
                p2m_goal_abs[objects_x[b]][objects_y[b]][message] += 1

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

for message in d_m2h:
    # object distribution
    d_m2h[message]     = d_m2h[message][:, :, :, :, :, :].sum(-1).sum(-1)
    d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, :, :].sum(-1).sum(-1)

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

    # ball or box distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, 2:4, :].sum(-1).sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, 2:4, :].sum(-1).sum(-1)

# goal distribution
#d_m2h     = d_m2h_goal
#d_m2h_abs = d_m2h_goal_abs

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

def imshow2(h, t="", add_agent=False, cbar=True):
    ax = plt.gca()
    ax.set_title(t, fontsize=10)
    norm = plt.Normalize(0.0, h.max())
    rgba = expand(cmap(norm(h)))
    if add_agent:
        rgba = addagent(rgba)
    im = ax.imshow(h, cmap=cmap, vmin=0.0, vmax=h.max())
    ax.imshow(rgba)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=str(100/N) + "%", pad=C/2)
        cbarr = fig.colorbar(im, ax=ax, cax=cax, orientation="vertical", ticks=[0, h.max()])
        cbarr.ax.set_yticklabels([r"$\mathrm{0.00}$", r"$\mathrm{" + "{:1.2f}".format(h.max()) + "}$"])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis("off")

### FIGURE ###

cmap = plt.cm.cividis

for i in range(15):
    m = M[i]

    t = ""
    for s in m:
        t += chr(97+int(s))

    h = normalize(d_m2h[tuple(m)][0].sum(0))

    if env == "GoToObjCustom" or env == "GoToObjCustom1D2" or env == "GoToObjCustom1DK2":
        h *= 2

    N = 13
    C = 32/300 #= 32/(32/300)) = 300 ppi
    A_bottom = 1
    A_top    = 2
    A_left   = 1
    A_right  = 3.5
    W = N+1+A_left+A_right
    H = N+A_bottom+A_top

    fig = plt.figure(num=None, figsize=(W*C, H*C))

    imshow2(h, r"$P(b | \texttt{" + t +"})$", add_agent=True, cbar=True)

    fig.subplots_adjust(bottom =     A_bottom/H,
                        top    = 1 - A_top   /H,
                        left   =     A_left  /W,
                        right  = 1 - A_right /W)
    
    if not os.path.exists("renders"):
        os.mkdir("renders")
    plt.savefig("renders/analysis_bm_" + env + "_n" + str(n_frames) + "_m_" + str(i) + ".pdf", dpi=32/C*3)

    size = fig.get_size_inches()

plt.show()
