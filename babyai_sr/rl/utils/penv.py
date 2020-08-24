from multiprocessing import Process, Pipe
import gym

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

def worker(conn, env, n):
    while True:
        cmd, action, prev_result = conn.recv()
        if cmd == "step":
            if prev_result[0]:
                # receiver's frame
                obs, reward, done, info = env.step(action)
                done = done or 64 <= env.step_count
                if done:
                    obs = env.reset()
                active = env.step_count % n != 0
                globs = obs.copy()
                globs["image"]   = get_global(env, obs)
                obs["image"]     = get_local(obs)
            else:
                # sender's frame
                reward = 0.0
                done   = False
                active = True
                obs    = prev_result[2]
                globs  = prev_result[1]
                if 3 < len(prev_result):
                    info   = prev_result[5]
                else:
                    info   = None
            conn.send((active, globs, obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            active = env.step_count % n != 0
            globs = obs.copy()
            globs["image"]   = get_global(env, obs)
            obs["image"]     = get_local(obs)
            conn.send((active, globs, obs))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env, n):
        assert len(env) >= 1, "No environment given."

        self.env             = env
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
        active = self.env[0].step_count % self.n != 0
        globs = obs.copy()
        globs["image"]   = get_global(self.env[0], obs)
        obs["image"]     = get_local(obs)
        self.prev_results = [(active, globs, obs)] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def step(self, actions):
        for local, action, prev_result in zip(self.locals, actions[1:], self.prev_results[1:]):
            local.send(("step", action, prev_result))
        if self.prev_results[0][0]:
            # receiver's frame
            obs, reward, done, info = self.env[0].step(actions[0])
            done = done or 64 <= self.env[0].step_count
            if done:
                obs = self.env[0].reset()
            active = self.env[0].step_count % self.n != 0
            globs = obs.copy()
            globs["image"]   = get_global(self.env[0], obs)
            obs["image"]     = get_local(obs)
        else:
            # sender's frame
            reward = 0.0
            done   = False
            active = True
            obs    = self.prev_results[0][2]
            globs  = self.prev_results[0][1]
            if 3 < len(self.prev_results[0]):
                info   = self.prev_results[0][5]
            else:
                info   = None
        self.prev_results = [(active, globs, obs, reward, done, info)] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()
