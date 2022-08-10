import os
import torch
import numpy as np
from .base import Base
from utils.utils import Normalize
from components.networks import ProbMLP
from components.memory import Memory
from utils.torch_truncnorm import TruncatedNormal

class Agent(Base):
    def __init__(self, cfg, env, device):

        # GENERAL PARAMS
        self.cfg = cfg
        self.env = env
        self.device = device
        self.normaliser = Normalize(self.env, cfg=self.cfg, device=device)
        self.obs_dim = len(self.normaliser.obs_space)
        self.act_dim = len(self.normaliser.act_space)
        self.state_act_dim = ((1 + cfg.window) * (self.obs_dim)) + self.act_dim
        self.n_steps = 0
        self.model_path = os.path.join(cfg.models_dir, 'model.pth')
        self.exploration_steps = self.cfg.exploration_mins / self.cfg.mins_per_step
        self.output_norm_range = [-1, 1]
        self.output_norm_low = torch.tensor([np.min(self.output_norm_range)], dtype=torch.float).to(self.device)
        self.output_norm_high = torch.tensor([np.max(self.output_norm_range)], dtype=torch.float).to(self.device)

        # COMPONENTS
        self.memory = Memory(cfg=self.cfg, obs_dim=self.obs_dim)
        self.cem_init_mean = torch.zeros(size=(cfg.horizon, self.act_dim),
                                         dtype=torch.float, requires_grad=False).to(self.device)
        self.cem_init_var = torch.tile(torch.tensor(cfg.init_var, requires_grad=False),
                                       (cfg.horizon, self.act_dim)).to(self.device)
        self.model_1 = ProbMLP(cfg, input_dims=self.state_act_dim, output_dims=self.obs_dim, device=device)
        self.model_2 = ProbMLP(cfg, input_dims=self.state_act_dim, output_dims=self.obs_dim, device=device)
        self.model_3 = ProbMLP(cfg, input_dims=self.state_act_dim, output_dims=self.obs_dim, device=device)
        self.model_4 = ProbMLP(cfg, input_dims=self.state_act_dim, output_dims=self.obs_dim, device=device)
        self.model_5 = ProbMLP(cfg, input_dims=self.state_act_dim, output_dims=self.obs_dim, device=device)
        self.ensemble = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5]
        self.model_idxs = [np.arange(i, cfg.particles, len(self.ensemble)) for i in range(len(self.ensemble))]

        # REWARD PARAMS
        temp_idx = []
        for temp in self.cfg.temp_reward:
            idx = self.normaliser.obs_space.index(temp)
            temp_idx.append(idx)
        self.temp_idx = temp_idx
        self.energy_idx = self.normaliser.obs_space.index(self.cfg.energy_reward)
        self.c02_idx = self.normaliser.obs_space.index(self.cfg.c02_reward)

        # EXPLORATION PARAMS
        lower = self.normaliser.action_lower_bound
        upper = self.normaliser.action_upper_bound
        self.expl_deltas = {}
        for key in lower.keys() & upper.keys():
            delta = (upper[key] - lower[key]) * self.cfg.expl_del
            self.expl_deltas[key] = delta

        super().__init__(env=self.env, normaliser=self.normaliser, memory=self.memory, cfg=self.cfg,
                         act_dim=self.act_dim, obs_space=self.normaliser.obs_space,
                         n_steps=self.n_steps, expl_deltas=self.expl_deltas)

    @torch.no_grad()
    def trajectory_sampler(self, init_state, act_seqs, planning_test=False):
        '''
        Takes action sequences and propogates each P times through models to horizon H.
        :param init_state: numpy array of most recent observation
        :param act_seqs: tensor of action sequences to be evaluated
        :return particle state trajectories: array of trajectories of shape: (particles, popsize, horizon, obs_dim + 4)
        '''
        # detach action sequences from computational graph
        if not planning_test:
            act_seqs = act_seqs.clone().cpu().detach().numpy()

        particle_act_seq = np.tile(act_seqs, (self.cfg.particles, 1, 1, 1))  # duplicate action sequence by no particles
        # convert state to tensor and duplicate

        if planning_test:
            state_tile = np.tile(init_state, (self.cfg.particles, 1, 1))  # duplicate state
        else:
            state_tile = np.tile(init_state, (self.cfg.particles, self.cfg.population, 1))  # duplicate state
        # instantiate trajectory tensor
        if planning_test:
            trajs = np.zeros(
                shape=(self.cfg.particles, 1, self.cfg.horizon, self.obs_dim + self.cfg.time_dim))

        else:
            trajs = np.zeros(
                shape=(self.cfg.particles, self.cfg.population, self.cfg.horizon, self.obs_dim))

        for i in range(self.cfg.horizon):
            action = particle_act_seq[:, :, i, :]
            trajs[:, :, i, :] = state_tile
            input = np.concatenate((action, state_tile, self.memory.previous_sampled), axis=2)

            # store state in memory after input has been created
            self.memory.store_previous_samples(state_tile)

            for j, model in enumerate(self.ensemble):
                model_input = input[self.model_idxs[j]]  # select subset of data for model
                model.float()  # ensure model parameters are floats
                model_input_T = torch.tensor(model_input, dtype=torch.float, requires_grad=False).to(self.device)

                mean_, var_ = model.forward(model_input_T)

                state_ = TruncatedNormal(loc=mean_, scale=var_, a=-2,
                                         b=2).sample()  # a and b stop discontinuity at -1,1

                # ensure state is in normalised region [-1,1]
                state_ = torch.where(state_ < self.output_norm_low, self.output_norm_low, state_)
                state_ = torch.where(state_ > self.output_norm_high, self.output_norm_high, state_)

                # state_ = self.normaliser.update_time(state_tensor=state_, init_date=self.TS_init_date,
                #                                      init_time=self.TS_init_time, TS_step=i)

                state_tile[self.model_idxs[j]] = state_.cpu().detach().numpy()

        return trajs  # remove time dim

    @torch.no_grad()
    def plan(self, observation, env):
        '''
        Selects action given current state either by Maximum Variance (MV) during system ID period or by maximum
        expected reward during normal control.
        :param observation: dict output of simulation describing current state of environment
        :param env: environment instance used to get current date and time
        :return action_dict: dictionary describing agent's current best estimate of the optimal action given state
        :return state_action: input to model, returned for storing in memory
        :return obs: normalised obs array, returned for storing in memory
        '''
        global action_dict
        min, hour, day, month = env.get_date()
        date = (min, hour, day, month)
        self.normaliser.month = month
        obs = self.normaliser.outputs(observation, date, for_memory=True)

        self.TS_init_time = (min, hour)
        self.TS_init_date = (day, month)

        self.memory.previous_sampled = np.tile(self.memory.previous, (self.cfg.particles, self.cfg.population, 1))

        # MPPI
        mean, var, t = self.cem_init_mean, self.cem_init_var, 0
        # cem optimisation loop
        while (t < self.cfg.max_iters) and (torch.max(var) > self.cfg.epsilon):
            dist = TruncatedNormal(loc=mean, scale=var, a=-2, b=2)  # range [-2,2] to avoid discontinuity at [-1,1]
            act_seqs = dist.sample(sample_shape=[self.cfg.population, ]).float()
            act_seqs = torch.where(act_seqs < torch.tensor([-1], dtype=torch.float, device=self.device),
                                   torch.tensor([-1], dtype=torch.float, device=self.device),
                                   act_seqs)  # clip
            act_seqs = torch.where(act_seqs > torch.tensor([1], dtype=torch.float, device=self.device),
                                   torch.tensor([1], dtype=torch.float, device=self.device),
                                   act_seqs)  # clip

            exp_rewards, exp_var = self.estimate_reward(obs, act_seqs)

            # maximise variance during system ID and expected reward otherwise
            if self.n_steps <= self.exploration_steps:
                target = exp_var
            else:
                target = exp_rewards

            elite_values = target[np.argsort(target)][-int(self.cfg.elites * self.cfg.population):]
            elite_actions = act_seqs[np.argsort(target)][-int(self.cfg.elites * self.cfg.population):]

            # update parameters
            max_value = target.max(0)[0]
            min_value = target.min(0)[0]
            norm_values = (np.absolute(elite_values) - np.absolute(min_value)) / \
                          (np.absolute(max_value) - np.absolute(min_value)) - 1  # scales to range [-1, 0]
            omega = torch.exp(self.cfg.temperature * norm_values).view(norm_values.shape[0], 1, 1).to(self.device)
            omega_tile = torch.tile(omega, (1, self.cfg.horizon, self.act_dim)).to(self.device)

            mean_ = torch.sum(omega_tile * elite_actions, dim=0) / (omega.sum(0) + 1e-9)
            var_ = torch.sqrt(torch.sum(omega_tile * (elite_actions - mean_.unsqueeze(0)) ** 2, dim=0) / \
                   (omega.sum(0) + 1e-9))

            mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * mean_
            var = self.cfg.momentum * var + (1 - self.cfg.momentum) * var_

            t += 1

            actions = mean[0].cpu().detach().numpy()  # first action is trajectory
            action_dict = self.normaliser.revert_actions(actions)
            state_action = np.concatenate((actions, obs, self.memory.previous))

        self.memory.store_previous(obs)  # store observation in model memory

        return action_dict, state_action, obs

    def estimate_reward(self, init_state: np.array, act_seqs: torch.tensor):
        '''
        Passes action sequences through trajectory sampler, then calculates expected reward and variance.
        :param init_state: Tensor of initial, normalised state of environment of shape (obs_dim,)
        :param act_seqs: Tensor of candidate action sequences of shape (popsize, horizon)
        :return rewards: Tensor of expected reward for each action sequence of shape (popsize,)
        :return var: Tensor of reward variance for each action sequence of shape (popsize,)
        '''

        particle_trajs = self.trajectory_sampler(init_state, act_seqs)
        particle_trajs_revert = self.normaliser.model_predictions_to_tensor(particle_trajs)  # drop time dim
        rewards, var = self.planning_reward(particle_trajs_revert)

        return rewards, var

    def planning_reward(self, particle_trajs: torch.tensor):
        '''
        Takes particles trajectories and calculates expected reward for each action sequence
        :param particle_trajs: Tensor of sampled particle trajectories of shape: (particles, popsize, horizon, obs_dim)
        :return exp_reward: Tensor of expected rewards for each action trajectory in popsize. Shape: (popsize,)
        : return var_reward: Tensor of reward variance for each action trajectory in popsize. Shape: (popsize,)
        '''
        energy_elements = particle_trajs[:, :, :, self.energy_idx]
        temp_elements = particle_trajs[:, :, :, self.temp_idx]

        temp_penalties = torch.minimum((self.cfg.low_temp_goal - temp_elements) ** 2,
                                       (self.cfg.high_temp_goal - temp_elements) ** 2) * -self.cfg.theta
        temp_rewards = torch.where(
            (self.cfg.low_temp_goal >= temp_elements) | (self.cfg.high_temp_goal <= temp_elements),
            temp_penalties, torch.tensor([0.0], dtype=torch.float))  # zero if in correct range, penalty otherwise
        temp_sum = torch.sum(temp_rewards, axis=[2, 3])  # sum across sensors and horizon

        c02_elements = particle_trajs[:, :, :, self.c02_idx]
        energy_elements_kwh = (energy_elements * (self.cfg.mins_per_step / 60)) / 1000
        c02 = c02_elements * energy_elements_kwh
        c02_sum = torch.sum(c02, axis=2)

        particle_rewards = temp_sum + c02_sum
        exp_reward = torch.mean(particle_rewards, axis=0)
        var_reward = torch.var(particle_rewards, axis=0)

        return exp_reward, var_reward

    def learn(self):
        '''
        Updates parameters of each learning dynamical model in ensemble and returns final losses
        :return mean_loss: mean loss across models
        '''

        losses = []

        for j, model in enumerate(self.ensemble):
            print('...updating ensemble model:', j, '...')

            state_action_arr, obs_array, batches = self.memory.sample()

            # if multiple sample update model min max logvar
            if obs_array.shape[0] > 1:
                var = np.var(obs_array, axis=0)
                var = np.where(var == 0, np.min(var[np.nonzero(var)]), var)  # replace zeroes with minimum var
                logvar = torch.tensor(np.log(var), dtype=torch.double).to(self.device)
                model.max_logvar = torch.where(logvar > model.max_logvar, logvar, model.max_logvar)
                model.min_logvar = torch.where(logvar < model.min_logvar, logvar, model.min_logvar)

            for i in range(self.cfg.epochs):
                if i % 10 == 0:
                    print('learning epoch:', i)

                batch_loss = []
                for batch in batches:
                    state_action_T = torch.tensor(state_action_arr[batch], dtype=torch.float).to(self.device)
                    obs_T = torch.tensor(obs_array[batch], dtype=torch.float).to(self.device)

                    model.optimizer.zero_grad()
                    mu_pred, var_pred = model.forward(state_action_T)
                    loss = model.loss(mu_pred, var_pred, obs_T)
                    loss.backward()
                    model.optimizer.step()

                    MSE = torch.nn.MSELoss()
                    mse_loss = MSE(mu_pred, obs_T)
                    batch_loss.append(mse_loss.cpu().detach().numpy())

                # log loss at final epoch
                if i == (self.cfg.epochs - 1):
                    losses.append(np.mean(batch_loss))

        if self.n_steps > self.exploration_steps:
            self.memory.clear()

        # mean MSE across models
        mean_loss = np.mean(losses)

        return mean_loss

    def save_models(self, models_dir: str):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        model_1_path = os.path.join(models_dir, 'model_1.pth')
        model_2_path = os.path.join(models_dir, 'model_2.pth')
        model_3_path = os.path.join(models_dir, 'model_3.pth')
        model_4_path = os.path.join(models_dir, 'model_4.pth')
        model_5_path = os.path.join(models_dir, 'model_5.pth')

        paths = [model_1_path, model_2_path, model_3_path, model_4_path, model_5_path]

        for i, model in enumerate(self.ensemble):
            model.save_checkpoint(paths[i])

    def load_models(self, paths: list = None):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        for i, model in enumerate(self.ensemble):
            model.load_checkpoint(paths[i])
