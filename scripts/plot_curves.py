import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["text.usetex"]         = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{times}\usepackage{latexsym}"

fig = plt.figure(num=None, figsize=(0.1/2+3.1+0.5/2+0.1/2, 0.1/4+(4-1)*0.5/4+2.325+0.1/4))

linestyles = [
              ("sixty-four", (0, (7, 1))),
              ("eight",      (0, (1, 3))),
              ("four",       (0, (3, 1))),
              ("two",        (0, (3, 1, 1, 1, 1, 1))),
              ("one",        (0, (1, 1))),
              ("arch",       (0, (7, 1, 3, 1, 3, 1))),
             ]

def plotter(i, color, linestyle="solid", label=None, start=0, add=0, multiplier=1, alpha=0.5, show_std=False):
    L = len(frames_means[i])
    
    start_episodes = 0
    if 0 < start:
        start_episodes = framess[i][start-1]
    start_episodes += add

    y = np.convolve(np.asarray(frames_means[i]), np.ones(2*k+1)/(2*k+1))[start+2*k:L]

    for style in linestyles:
        if linestyle == style[0]:
            linestyle = style[1]
            break

    if label is None:
        plt.plot(np.array(framess[i][start + k:L - k]) - start_episodes, multiplier*y, color=color, alpha=alpha, linestyle=linestyle)
    else:
        plt.plot(np.array(framess[i][start + k:L - k]) - start_episodes, multiplier*y, color=color, alpha=alpha, linestyle=linestyle, label=label)

    if show_std:
        std = np.sqrt(np.convolve(np.asarray(frames_stds[i])**2, np.ones(2*k + 1)/(2*k + 1)**2)[start + 2*k:L])
        plt.fill_between(np.array(framess[i][start + k:L - k]) - start_episodes, multiplier*(y + std), multiplier*(y - std), color=color, alpha=0.1*alpha)

################################################################################
### GLOBAL SETTINGS ############################################################

# take running average over 2*k + 1 batches
k = 37

FONTSIZE = 10

################################################################################
### READ LOGS ##################################################################

framess      = []
frames_means = []
frames_stds  = []
for file in [
             "logs/GoToObj/1/log.csv",                         #  0
             "logs/GoToObj/2/log.csv",                         #  1
             "logs/GoToObj/4/log.csv",                         #  2
             "logs/GoToObj/8/log.csv",                         #  3
             "logs/GoToObj/64/log.csv",                        #  4
             "logs/GoToObj/arch/log.csv",                      #  5
             "logs/GoToObj/no-comm/log.csv",                   #  6
             
             "logs/GoToObjUnlocked/1/log.csv",                 #  7
             "logs/GoToObjUnlocked/2/log.csv",                 #  8
             "logs/GoToObjUnlocked/4/log.csv",                 #  9
             "logs/GoToObjUnlocked/8/log.csv",                 # 10
             "logs/GoToObjUnlocked/64/log.csv",                # 11
             "logs/GoToObjUnlocked/arch/log.csv",              # 12
             "logs/GoToObjUnlocked/no-comm/log.csv",           # 13
             
             "logs/GoToObjLocked/1/log.csv",                   # 14
             "logs/GoToObjLocked/2/log.csv",                   # 15
             "logs/GoToObjLocked/4/log.csv",                   # 16
             "logs/GoToObjLocked/8/log.csv",                   # 17
             "logs/GoToObjLocked/64/log.csv",                  # 18
             "logs/GoToObjLocked/arch/log.csv",                # 19
             "logs/GoToObjLocked/no-comm/log.csv",             # 20
             
             "logs/GoToObjLocked_(ambiguous)/1/log.csv",       # 21
             "logs/GoToObjLocked_(ambiguous)/2/log.csv",       # 22
             "logs/GoToObjLocked_(ambiguous)/4/log.csv",       # 23
             "logs/GoToObjLocked_(ambiguous)/8/log.csv",       # 24
             "logs/GoToObjLocked_(ambiguous)/64/log.csv",      # 25
             "logs/GoToObjLocked_(ambiguous)/arch/log.csv",    # 26
             "logs/GoToObjLocked_(ambiguous)/no-comm/log.csv", # 27
             
             "logs/GoToObjLocked/1/pretrained/GoToObjUnlocked/1/log.csv",                       # 28
             "logs/GoToObjLocked/1/pretrained/GoToObjUnlocked/no-comm/log.csv",                 # 29
             "logs/GoToObjLocked/64/pretrained/GoToObjUnlocked/64/log.csv",                     # 30
             "logs/GoToObjLocked/64/pretrained/GoToObjUnlocked/no-comm/log.csv",                # 31
             "logs/GoToObjLocked/arch/pretrained/GoToObjUnlocked/arch/log.csv",                 # 32
             "logs/GoToObjLocked/no-comm/pretrained/GoToObjUnlocked/no-comm/log.csv",           # 33
             
             "logs/GoToObjLocked_(ambiguous)/1/pretrained/GoToObjLocked/1/log.csv",             # 34
             "logs/GoToObjLocked_(ambiguous)/1/pretrained/GoToObjLocked/no-comm/log.csv",       # 35
             "logs/GoToObjLocked_(ambiguous)/64/pretrained/GoToObjLocked/64/log.csv",           # 36
             "logs/GoToObjLocked_(ambiguous)/64/pretrained/GoToObjLocked/no-comm/log.csv",      # 37
             "logs/GoToObjLocked_(ambiguous)/arch/pretrained/GoToObjLocked/arch/log.csv",       # 38
             "logs/GoToObjLocked_(ambiguous)/no-comm/pretrained/GoToObjLocked/no-comm/log.csv", # 39
            ]:
    with open(file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        frames      = []
        frames_mean = []
        frames_std  = []
        for i, row in enumerate(csvreader):
            if i != 0:
                frames.append(int(row[1]))
                frames_mean.append(float(row[10]))
                frames_std.append(float(row[11]))
        framess.append(frames)
        frames_means.append(frames_mean)
        frames_stds.append(frames_std)

plt.title(r"$\textrm{Placeholder Title,}$ $|\mathcal{M}| = |\mathcal{V}|^k$", fontsize=FONTSIZE)

y_low = 5

################################################################################
### PLOT LINES #################################################################

#title = r"$\textrm{GoToObj,}$ $|\mathcal{M}| = 8^{8}$"
#plotter(6,  'r',                         label=r"$\textrm{no}$ $\textrm{comm.}$")
#plotter(4,  'y', linestyle="sixty-four", label=r"$n = 64$")
#plotter(3,  'g', linestyle="eight",      label=r"$n = 8$")
#plotter(2,  'c', linestyle="four",       label=r"$n = 4$")
#plotter(1,  'b', linestyle="two",        label=r"$n = 2$")
#plotter(0,  'm', linestyle="one",        label=r"$n = 1$")
#plotter(5,  'k', linestyle="arch",       label=r"$\textrm{Arch.}$ $\textrm{receiver}$")

#title = r"$\textrm{GoToObjUnlocked,}$ $|\mathcal{M}| = 8^{8}$"
#plotter(13, 'r',                         label=r"$\textrm{no}$ $\textrm{comm.}$")
#plotter(11, 'y', linestyle="sixty-four", label=r"$n = 64$")
#plotter(10, 'g', linestyle="eight",      label=r"$n = 8$")
#plotter(9,  'c', linestyle="four",       label=r"$n = 4$")
#plotter(8,  'b', linestyle="two",        label=r"$n = 2$")
#plotter(7,  'm', linestyle="one",        label=r"$n = 1$")
#plotter(12, 'k', linestyle="arch",       label=r"$\textrm{Arch.}$ $\textrm{receiver}$")

title = r"$\textrm{GoToObjLocked,}$ $|\mathcal{M}| = 8^{8}$"
plotter(20, 'r',                         label=r"$\textrm{no}$ $\textrm{comm.}$")
plotter(18, 'y', linestyle="sixty-four", label=r"$n = 64$")
plotter(17, 'g', linestyle="eight",      label=r"$n = 8$")
plotter(16, 'c', linestyle="four",       label=r"$n = 4$")
plotter(15, 'b', linestyle="two",        label=r"$n = 2$")
plotter(14, 'm', linestyle="one",        label=r"$n = 1$")
plotter(19, 'k', linestyle="arch",       label=r"$\textrm{Arch.}$ $\textrm{receiver}$")
y_low = 10

#title = r"$\textrm{GoToObjLocked (ambiguous),}$ $|\mathcal{M}| = 8^{8}$"
#plotter(27, 'r',                         label=r"$\textrm{no}$ $\textrm{comm.}$")
#plotter(25, 'y', linestyle="sixty-four", label=r"$n = 64$")
#plotter(24, 'g', linestyle="eight",      label=r"$n = 8$")
#plotter(23, 'c', linestyle="four",       label=r"$n = 4$")
#plotter(22, 'b', linestyle="two",        label=r"$n = 2$")
#plotter(21, 'm', linestyle="one",        label=r"$n = 1$")
#plotter(26, 'k', linestyle="arch",       label=r"$\textrm{Arch.}$ $\textrm{receiver}$")
#y_low = 10

#title = r"\noindent $\textrm{GoToObjLocked,}$ $\textrm{pretrained}$ $\textrm{with}$\\$\textrm{GoToObjUnlocked,}$ $|\mathcal{M}| = 8^{8}$"
#plotter(33, 'r', start=2000,                         label=r"$\textrm{no}$ $\textrm{comm.,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{comm.}$")
#plotter(31, 'y', start=2000, linestyle="sixty-four", label=r"$n = 64\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{comm.}$")
#plotter(30, 'g', start=2400, linestyle="eight",      label=r"$n = 64\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $n = 64$")
#plotter(28, 'b', start=3400, linestyle="two",        label=r"$n = 1\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $n = 1$")
#plotter(29, 'm', start=2000, linestyle="one",        label=r"$n = 1\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{comm.}$")
#plotter(32, 'k', start=2750, linestyle="arch",       label=r"$\textrm{Arch.}$ $\textrm{receiver,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{Arch.}$ $\textrm{receiver}$")
#y_low = 10

#title = r"\noindent $\textrm{GoToObjLocked (ambiguous),}$ $\textrm{pretrained}$ $\textrm{with}$\\$\textrm{GoToObjLocked,}$ $|\mathcal{M}| = 8^{8}$"
#plotter(39, 'r',                         start=2700, label=r"$\textrm{no}$ $\textrm{communication,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{communication}$")
#plotter(37, 'y', linestyle="sixty-four", start=2700, label=r"$n = 64\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{communication}$")
#plotter(36, 'g', linestyle="eight",      start=2700, label=r"$n = 64\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $n = 64$")
#plotter(34, 'b', linestyle="two",        start=4100, label=r"$n = 1\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $n = 1$")
#plotter(35, 'm', linestyle="one",        start=2700, label=r"$n = 1\textrm{,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{no}$ $\textrm{communication}$")
#plotter(38, 'k', linestyle="arch",       start=2950, label=r"$\textrm{Arch.}$ $\textrm{receiver,}$ $\textrm{pretrained}$ $\textrm{with}$ $\textrm{Arch.}$ $\textrm{receiver}$")
#y_low = 10

################################################################################
### PLOT REST ##################################################################

ax = plt.gca()

# x-axis
plt.xlabel(r"$\textrm{episodes}$", fontsize=FONTSIZE)
plt.xscale("log")
plt.xlim(920, 2800000)
ax.set_xticks(     [     1000,     10000,    100000,   1000000])
ax.set_xticklabels([r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"], fontsize=FONTSIZE)

# y-axis
plt.ylabel(r"$\textrm{time}$ $\textrm{steps}$ $\textrm{per}$ $\textrm{episode}$", fontsize=FONTSIZE)
plt.yscale("log")
if y_low == 10:
    plt.ylim(0.96*10, 68)
    ax.set_yticks(     [     10,      20,      40])
    ax.set_yticklabels([r"$10$", r"$20$", r"$40$"], fontsize=FONTSIZE)
    ax.set_yticks(     [ 30,  50,  60], minor=True)
    ax.set_yticklabels([r"", r"", r""], minor=True, fontsize=FONTSIZE)
else:
    plt.ylim(0.96*5, 68)
    ax.set_yticks(     [     5,      10,      20,      40])
    ax.set_yticklabels([r"$5$", r"$10$", r"$20$", r"$40$"], fontsize=FONTSIZE)

plt.legend(fontsize=9, handlelength=2.95*0.9, loc="lower left")

plt.tight_layout()

plt.title(title, fontsize=FONTSIZE)

plt.savefig("plot.pdf")

plt.show()
