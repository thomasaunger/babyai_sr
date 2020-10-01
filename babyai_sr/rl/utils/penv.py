import gym
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from multiprocessing import Process, Pipe

def get_global(env, obs):
    # get global view
    grid = env.grid
    
    # position agent
    x, y = env.agent_pos
    
    # rotate to match agent's orientation
    for i in range(env.agent_dir + 1):
        # rotate grid
        grid = grid.rotate_left()
        
        # rotate position of agent
        x_new = y
        y_new = grid.height - 1 - x
        x     = x_new
        y     = y_new
    
    # encode image for model
    image = grid.encode()

    # overlap global with receiver observation, i.e., include carried objects
    image[x, y, :] = obs["image"][3, 6, :]
    
    # indicate position of agent
    image[x, y, 0] += 10
    
    return image

def get_local(obs):
    # get local view
    return obs["image"][3:4, 5:7, :]

def get_agent_loc(env):
    # get global view
    grid = env.grid
    
    # position agent
    x, y = env.agent_pos
    
    # rotate to match agent's orientation
    for i in range(env.agent_dir + 1):
        # rotate position of agent
        x_new = y
        y_new = grid.height - 1 - x
        x     = x_new
        y     = y_new
    
    agent_x = x
    agent_y = y
    
    return agent_x, agent_y

def get_goal(env):
    goal_type  = OBJECT_TO_IDX[env.instrs.desc.type]
    goal_color = COLOR_TO_IDX[env.instrs.desc.color]
    
    return goal_type, goal_color

def get_goal_loc(globs):
    x, y = (( 3 < globs["image"][:, :, 0]) * (globs["image"][:, :, 0] <  8) +
            (13 < globs["image"][:, :, 0]) * (globs["image"][:, :, 0] < 18)).nonzero()
    
    goal_x = x[0]
    goal_y = y[0]
    
    return goal_x, goal_y

def worker(conn, env, n):
    while True:
        cmd, action, prev_result = conn.recv()
        if cmd == "step":
            if prev_result[0][1]:
                # receiver's frame
                obs, reward, done, info = env.step(action)
                done = done or 64 <= env.step_count
                if done:
                    obs = env.reset()
                active_sender = env.step_count % n == 0
                active_receiver = not active_sender
                active = (active_sender, active_receiver)
                sending = (active_sender, False)
                globs = obs.copy()
                globs["image"] = get_global(env, obs)
                obs["image"]   = get_local(obs)
                agent_x, agent_y      = get_agent_loc(env)
                goal_type, goal_color = get_goal(env)
                goal_x, goal_y        = get_goal_loc(globs)
                extra = (agent_x, agent_y, goal_type, goal_color, goal_x, goal_y)
            else:
                # sender's frame
                reward = 0.0
                done   = False
                active_sender = False
                active_receiver = not active_sender
                active = (active_sender, active_receiver)
                sending = (active_sender, False)
                obs    = prev_result[3]
                globs  = prev_result[2]
                extra  = prev_result[4]
            conn.send((active, sending, globs, obs, extra, reward, done))
        elif cmd == "reset":
            obs = env.reset()
            active_sender = env.step_count % n == 0
            active_receiver = not active_sender
            active = (active_sender, active_receiver)
            sending = (active_sender, False)
            globs = obs.copy()
            globs["image"] = get_global(env, obs)
            obs["image"]   = get_local(obs)
            agent_x, agent_y      = get_agent_loc(env)
            goal_type, goal_color = get_goal(env)
            goal_x, goal_y        = get_goal_loc(globs)
            extra = (agent_x, agent_y, goal_type, goal_color, goal_x, goal_y)
            conn.send((active, sending, globs, obs, extra))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env, n):
        assert len(env) >= 1, "No environment given."

        self.env             = env
        self.num_procs       = len(env)
        self.n               = n
        self.observation_space = self.env[0].observation_space
        self.action_space = self.env[0].action_space
        
        self.locals = []
        self.processes = []
        for i, env in enumerate(self.env[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, n))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None, None))
        obs = self.env[0].reset()
        active_sender = self.env[0].step_count % self.n == 0
        active_receiver = not active_sender
        active = (active_sender, active_receiver)
        sending = (active_sender, False)
        globs = obs.copy()
        globs["image"]   = get_global(self.env[0], obs)
        obs["image"]     = get_local(obs)
        agent_x, agent_y      = get_agent_loc(self.env[0])
        goal_type, goal_color = get_goal(self.env[0])
        goal_x, goal_y        = get_goal_loc(globs)
        extra = (agent_x, agent_y, goal_type, goal_color, goal_x, goal_y)
        self.prev_results = [(active, sending, globs, obs, extra)] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def step(self, actions):
        for local, action, prev_result in zip(self.locals, actions[1:, 1], self.prev_results[1:]):
            local.send(("step", action, prev_result))
        if self.prev_results[0][0][1]:
            # receiver's frame
            obs, reward, done, info = self.env[0].step(actions[0, 1])
            done = done or 64 <= self.env[0].step_count
            if done:
                obs = self.env[0].reset()
            active_sender = self.env[0].step_count % self.n == 0
            active_receiver = not active_sender
            active = (active_sender, active_receiver)
            sending = (active_sender, False)
            globs = obs.copy()
            globs["image"]   = get_global(self.env[0], obs)
            obs["image"]     = get_local(obs)
            agent_x, agent_y      = get_agent_loc(self.env[0])
            goal_type, goal_color = get_goal(self.env[0])
            goal_x, goal_y        = get_goal_loc(globs)
            extra = (agent_x, agent_y, goal_type, goal_color, goal_x, goal_y)
        else:
            # sender's frame
            reward = 0.0
            done   = False
            active_sender = False
            active_receiver = not active_sender
            active = (active_sender, active_receiver)
            sending = (active_sender, False)
            obs    = self.prev_results[0][3]
            globs  = self.prev_results[0][2]
            extra  = self.prev_results[0][4]
        self.prev_results = [(active, sending, globs, obs, extra, reward, done)] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()
