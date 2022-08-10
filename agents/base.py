import copy
import datetime
import numpy as np
import pandas as pd


class Base:
    '''
    General parent class that defines common model-based agent methods
    '''

    def __init__(self, env, normaliser, memory, cfg, act_dim, obs_space, n_steps, expl_deltas):
        self.env, self.normaliser, self.memory, self.cfg = env, normaliser, memory, cfg
        self.act_dim, self.obs_space, self.n_steps, self.expl_deltas = act_dim, obs_space, n_steps, expl_deltas

        if 'cO2_path' in cfg.keys():
            self.cO2_data = pd.read_pickle(cfg.cO2_path)

    def one_step_reward(self, state_dict):
        '''
        Calculates reward from dictionary output of environment
        :param state_dict: Dictionary defining the state of variables in an observation
        :return reward: Scalar reward
        '''

        temp_reward = 0
        for t in self.cfg.temp_reward:
            temp = state_dict[t]
            if (self.cfg.low_temp_goal <= temp) and (temp <= self.cfg.high_temp_goal):
                pass
            else:
                temp_reward -= self.cfg.theta * min((self.cfg.low_temp_goal - temp) ** 2,
                                                    (self.cfg.high_temp_goal - temp) ** 2)

        c02 = state_dict[self.cfg.c02_reward]  # gCO2/kWh
        energy_kwh = (state_dict[self.cfg.energy_reward] * (self.cfg.mins_per_step / 60)) / 1000  # kWh
        c02_reward = -(c02 * energy_kwh)  # gC02
        reward = c02_reward + temp_reward

        return reward

    def add_c02(self, observation_):
        '''
        Takes observation dictionary and adds C02 if include_grid == True
        :param observation_: dictionary of state of environment
        :return observation_:
        '''

        # do not update observation if it already contains C02 data
        if 'Grid_CO2' in self.obs_space:
            return observation_

        obs = copy.deepcopy(observation_)
        min, hour, day, month = self.env.get_date()

        dt = datetime.datetime(self.cfg.c02_year, month, day, hour, min)
        # index c02 data from dataframe using datetime of simulation
        c02 = \
         self.cO2_data[self.cfg.c02_carbon_col][self.cO2_data[self.cfg.c02_dt_col] == dt].values[0]
        obs['c02'] = c02

        return obs
