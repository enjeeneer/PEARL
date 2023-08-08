import wandb
import torch
import energym
import numpy as np
from tqdm import tqdm
from configs.parser import parse_cfg
from agents.pearl import Agent as PEARL

envs = ['OfficesThermostat-v0', 'MixedUseFanFCU-v0', 'SeminarcenterThermostat-v0']

if __name__ == '__main__':
    for env in envs:
        agent_cfg_path = 'configs/pearl.yaml'
        env_cfg_path = 'configs/envs.yaml'
        cfg = parse_cfg(agent_cfg_path, env_cfg_path, env_name=env)

        N = int((60 * 24) / cfg.mins_per_step)  # make model updates at end of each day
        steps_per_day = int((60 * 24) / cfg.mins_per_step)
        sim_steps = steps_per_day * cfg.days

        cfg.steps_per_day = steps_per_day
        cfg.include_grid = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # wandb setup (enter your own details here should you wish to use wandb)
        run = wandb.init(
            project='pearl',
            entity="enjeeneer",
            config=dict(cfg),
            reinit=True
        )
        wandb.config.update(dict(cfg))

        env = energym.make(cfg.env_name, weather=cfg.weather, simulation_days=cfg.days)
        agent = PEARL(cfg=cfg, env=env, device=device)

        learn_iters = 0
        daily_scores = []
        emissions = []
        temps = []

        obs = env.get_output()
        if agent.cfg.include_grid:
            obs = agent.add_c02(obs)
        score = 0
        for i in tqdm(range(sim_steps)):
            action_dict, state_action, last_obs = agent.plan(obs, env)
            obs_ = env.step(action_dict)
            if agent.cfg.include_grid:
                obs_ = agent.add_c02(obs_)
            reward = agent.one_step_reward(obs_)
            score += reward
            agent.n_steps += 1
            if agent.n_steps > cfg.window:  # skip first two state-action pairs
                agent.memory.store(state_action=state_action, obs=last_obs)

            emissions.append(
                obs_[agent.cfg.energy_reward] * (cfg.mins_per_step / 60) / 1000
                * (obs_[agent.cfg.c02_reward] / 1000)
            )
            temps.append(obs_['Z02_T'])

            min, hour, day, month = env.get_date()
            merged_data = {**obs_}
            wandb.log(merged_data)

            # exploration phase update
            if (agent.n_steps <= agent.cfg.exploration_steps) and (agent.n_steps > cfg.window):
                model_loss = agent.learn()
                learn_iters += 1

            # normal update
            if agent.n_steps % cfg.steps_per_day == 0:
                ### UPDATE ###
                model_loss = agent.learn()
                learn_iters += 1

                daily_scores.append(score)
                avg_score = np.mean(daily_scores[-3:])

                _, _, day, month = env.get_date()

                print('date:', day, '/', month, '--',
                      'today\'s score %.1f' % score, 'avg score %.1f' % avg_score,
                      'learning steps', learn_iters)

                wandb.log({'train/mean_zone_temp': np.mean(temps[-cfg.steps_per_day:]),
                           'train/emissions': sum(emissions),
                           'train/reward': score,
                           'train/model_loss': model_loss,
                           })

                score = 0

            obs = obs_

        run.finish()
        env.close()
