import numpy as np

class Memory:
    def __init__(self, cfg, obs_dim):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.state_actions = []
        self.obs_ = []
        self.previous = np.zeros(shape=(obs_dim * cfg.window,))
        self.mem_size = cfg.steps_per_day
        if cfg.deterministic:
            self.previous_sampled = np.zeros(shape=(cfg.popsize, obs_dim * cfg.window))
        else:
            self.previous_sampled = np.zeros(shape=(cfg.particles, cfg.population, obs_dim * cfg.window))
        self.mem_ctr = int(0)

    def sample(self):
        '''
        Generates batches of training data for dynamical model from previously executed state actions and observations
        :return: array of stored state actions of shape (datapoints, state_act_dim)
        :return: array of stored next observations of shape (datapoints, obs_dim)
        :return: array of batch indices of shape (datapoints/batch_size, batch_size)
        '''
        datapoints = len(self.state_actions[:-1])
        batch_start = np.arange(0, datapoints, self.cfg.batch_size)
        indices = np.random.choice(datapoints, size=datapoints, replace=True)
        batches = [indices[i:i + self.cfg.batch_size] for i in batch_start]

        return np.array(self.state_actions[:-1]), np.array(self.obs_[1:]), batches

    def store(self, state_action, obs):
        '''
        Stores state action and observation in memory.
        :param state_action: normalised array of state_actions of shape (state_act_dim,)
        :param observation: normalised array of obs at previous timestep of shape (observation,)
        '''
        self.state_actions.append(state_action)
        self.obs_.append(obs)

    def store_previous(self, state_tensor):
        '''
        Takes current state and stores in working memory for use in future action selection
        :param state_tensor:
        '''
        self.previous[:self.obs_dim] = self.previous[self.obs_dim:]
        self.previous[self.obs_dim:] = state_tensor

    def store_previous_samples(self, state_matrix):
        '''
        Stores states sampled using trajectory sampler (TS) in working memory for sampler propogation
        :param state_matrix: Tensor of states sampled using TS of shape (particles, popsize, obs_dim*past_window_size)
        :return:
        '''
        if self.cfg.deterministic:
            self.previous_sampled[:, :self.obs_dim] = self.previous_sampled[:, self.obs_dim:]
            self.previous_sampled[:, self.obs_dim:] = state_matrix

        else:
            self.previous_sampled[:, :, :self.obs_dim] = self.previous_sampled[:, :, self.obs_dim:]
            self.previous_sampled[:, :, self.obs_dim:] = state_matrix

    def clear(self):
        '''
        Clears working memory after each learning procedure.
        '''
        self.state_actions = []
        self.obs_ = []