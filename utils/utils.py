import torch
import datetime
import numpy as np
from copy import deepcopy
from energym.spaces.box import Box

class Normalize:
    def __init__(self, env, cfg, device):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.act_space = [key for key in env.get_inputs_names() if key not in cfg.redundant_actions]
        self.obs_space = [key for key in env.get_outputs_names() if key not in cfg.redundant_observations]
        self.month = 1

        # OUTPUT SPACE BOUNDS
        upper_bound = {}
        lower_bound = {}

        default_upper = {key: env.output_space[key].high[0] for key in self.obs_space}
        default_lower = {key: env.output_space[key].low[0] for key in self.obs_space}

        self.output_lower_bound = {**lower_bound, **default_lower}
        self.output_upper_bound = {**upper_bound, **default_upper}

        # add c02 if not already in energym environment
        if cfg.include_grid and 'Grid_CO2' not in self.obs_space:
            self.obs_space.append('c02')
            self.output_lower_bound['c02'] = self.cfg.c02_low
            self.output_upper_bound['c02'] = self.cfg.c02_high

        output_lower_bound_T = []
        output_upper_bound_T = []
        for key, value in self.output_lower_bound.items():
            output_lower_bound_T.append(value)

        for key, value in self.output_upper_bound.items():
            output_upper_bound_T.append(value)

        self.output_lower_bound_T = torch.tensor(output_lower_bound_T, requires_grad=False)
        self.output_upper_bound_T = torch.tensor(output_upper_bound_T, requires_grad=False)

        # ACTION SPACE BOUNDS
        cont_keys = [p for p in list(self.env.input_space.spaces.keys()) if isinstance(self.env.input_space[p], Box)]
        upper_bound = {}
        lower_bound = {}

        default_upper = {key: self.env.input_space[key].high[0] for key in cont_keys}
        default_lower = {key: self.env.input_space[key].low[0] for key in cont_keys}

        self.action_lower_bound = {**lower_bound, **default_lower}
        self.action_upper_bound = {**upper_bound, **default_upper}

    def outputs(self, outputs: dict, date, for_memory=False):
        '''
        Takes outputs on EnergyPlus simulation and normalises in [-1,1]
        :param outputs: dictionary of simulation outputs
        :param date: tuple of (min, hour, day, month) i.e. current datetime of simulation
        :param for_memory: boolean flag to indicate whether to add time features, which we only do if we
        aren't storing outputs in memory.
        :return output_arr: array normalised outputs of shape (obs_dim,)
        '''

        # get output keys
        output_cp = deepcopy(outputs)
        # drop redundant features
        if len(self.cfg.redundant_observations) > 0:
            for key in self.cfg.redundant_observations:
                if key in output_cp:
                    del output_cp[key]

        shared_keys = [p for p in output_cp if p in list(self.output_lower_bound.keys())]

        # normalise values
        for key in shared_keys:
            output_cp[key] = 2 * (output_cp[key] - self.output_lower_bound[key]) / (
                    self.output_upper_bound[key] - self.output_lower_bound[key]) - 1

        # add time and date features
        (min, hour, day, month) = date
        time = hour * (self.cfg.steps_per_day / 24) + (min / ((60 * 24) / self.cfg.steps_per_day))
        date_delta = datetime.date(2017, month, day) - datetime.date(2017, 1, 1)
        days = date_delta.days

        if not for_memory:  # we don't store time in memory because we don't predict it with models
            # transform time to sit on two dimensional (circular) sin/cos space
            output_cp['sin_time'] = np.sin(2 * np.pi * (time / self.cfg.steps_per_day))
            output_cp['cos_time'] = np.cos(2 * np.pi * (time / self.cfg.steps_per_day))

            # transform date to sit on two dimensional (circular) sin/cos space
            output_cp['sin_date'] = np.sin(2 * np.pi * (days / 365))
            output_cp['cos_date'] = np.cos(2 * np.pi * (days / 365))

        # convert to array-like
        output_arr = np.array(list(output_cp.values()), dtype=np.float).reshape(
            len(output_cp.values()), )  # extract values from dict

        return output_arr

    def model_predictions_to_tensor(self, outputs: np.array):
        '''
        Takes state prediction tensor from model in range [-1,1] and unnormalises to allow reward calculation.
        :param output_tensor: Tensor holding next state prediction of shape: (*,obs_dim)
        :return: Unnormalised tensor of shape: (*, obs_dim)
        '''
        return ((torch.tensor(outputs, dtype=torch.float, requires_grad=False) + 1) / 2) * (
                    self.output_upper_bound_T - self.output_lower_bound_T) + self.output_lower_bound_T

    def revert_actions(self, actions: np.array):
        '''
        Takes action selection and unnormalises as preparing for passing to energyplus simulation.
        :param actions: array of action sequences of shape (act_dim,)
        :return action dict: dictionary of action values
        '''

        actions_dict = {}
        actions_cp = deepcopy(actions)

        for i, key in enumerate(self.act_space):
            actions_dict[key] = actions_cp[i]

        # un-normalise values
        for key in self.cfg.continuous_actions:
            revert = ((actions_dict[key] + 1) / 2) * (self.action_upper_bound[key] - self.action_lower_bound[key]) \
                     + self.action_lower_bound[key]

            # clip actions into appropriate bounds
            actions_dict[key] = [np.clip(revert, a_min=self.action_lower_bound[key],
                                         a_max=self.action_upper_bound[key])]

        # manually interpret discrete elements of action array
        if len(self.cfg.discrete_actions) > 0:
            for key in self.cfg.discrete_actions:
                if actions_dict[key] < 0:
                    actions_dict[key] = [0]
                else:
                    actions_dict[key] = [1]

        # make sure heating and cooling are opposite in offices
        if self.cfg.env_name == 'OfficesThermostat-v0':
            if (self.month <= 5) or (self.month >= 11):
                actions_dict['Bd_Heating_onoff_sp'] = [1]
                actions_dict['Bd_Cooling_onoff_sp'] = [0]
            else:
                actions_dict['Bd_Heating_onoff_sp'] = [0]
                actions_dict['Bd_Cooling_onoff_sp'] = [1]

        return actions_dict

    def norm_actions(self, action_dict):
        '''
        Takes action dictionary and normalises so it can be stored in memory.
        :return actions: array of actions of shape (act_dims,)
        '''

        working_dict = deepcopy(action_dict)

        # normalise continuous actions
        for key in self.cfg.continuous_actions:
            working_dict[key][0] = 2 * (working_dict[key][0] - self.action_lower_bound[key]) / (
                    self.action_upper_bound[key] - self.action_lower_bound[key]) - 1

        # update action in office discrete action space
        for key in self.cfg.discrete_actions:
            val = working_dict[key][0]
            if val == 0:
                working_dict[key][0] = -1
            else:
                working_dict[key][0] = 1

        # drop redundant actions
        if len(self.cfg.redundant_actions) > 0:
            for key in self.cfg.redundant_actions:
                working_dict.pop(key)

        actions = np.array(list(working_dict.values()), dtype=float).reshape(len(working_dict.values()), )

        return actions

    def update_time(self, state_tensor: torch.tensor, init_date: tuple, init_time: tuple, TS_step: int):
        '''
        Takes state tensor as predicted by trajectory sampler and adds datetime features given the datestime at start
        of trajectory sampling and number of elapsed steps in sampler.
        :param state_tensor: tensor of sampled state predictions of shape [particles, popsize, obs_dim]
        :param init_time: tuple of (minute, hour)
        :param init_date: tuple of (day, month)
        :param TS_step: int in [0, pred_horizon]
        :return: Concat of state tensor and time tensor of shape [particles, popsize, obs_dim + time_dim]
        '''

        init_dtime = datetime.datetime(2017, init_date[1], init_date[0], init_time[1], init_time[0])
        step_dtime = init_dtime + datetime.timedelta(minutes=(24 * 60 / self.cfg.steps_per_day) * (TS_step + 1))

        # no. of timesteps elapsed in day and days elapsed in year
        time = step_dtime.hour * (self.cfg.steps_per_day / 24) + (
                    step_dtime.minute / ((60 * 24) / self.cfg.steps_per_day))  # timesteps elasped on this day
        date_delta = datetime.date(2017, step_dtime.month, step_dtime.day) - datetime.date(2017, 1, 1)
        days = date_delta.days

        # transform time to sit on two dimensional (circular) sin/cos space
        sin_time = np.sin(2 * np.pi * (time / self.cfg.steps_per_day))
        cos_time = np.cos(2 * np.pi * (time / self.cfg.steps_per_day))

        # transform date to sit on two dimensional (circular) sin/cos space
        sin_date = np.sin(2 * np.pi * (days / 365))
        cos_date = np.cos(2 * np.pi * (days / 365))

        # concat to tensor
        if self.cfg.name == 'pearl':
            time_tensor = torch.tile(torch.tensor([sin_time, cos_time, sin_date, cos_date], dtype=torch.float,
                                requires_grad=False), [state_tensor.shape[0], state_tensor.shape[1], 1]).to(self.device)

        elif self.cfg.name == 'mpc':
            time_tensor = torch.tile(torch.tensor([sin_time, cos_time, sin_date, cos_date], dtype=torch.float,
                                requires_grad=False), [state_tensor.shape[0], 1]).to(self.device)

        return torch.cat((state_tensor, time_tensor), dim=-1)