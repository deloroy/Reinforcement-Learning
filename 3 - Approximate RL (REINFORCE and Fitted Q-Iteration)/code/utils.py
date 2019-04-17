import numpy as np
import copy
from scipy import stats


def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1, render=False):
    paths = []

    for _ in range(n_episodes):
        observations = []
        actions = []
        rewards = []
        next_states = []

        state = mdp.reset()
        for _ in range(horizon):
            action = policy.draw_action(state)
            next_state, reward, terminal, _ = mdp.step(action)
            if render:
                mdp.render()
            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory

        paths.append(dict(
            states=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states)
        ))
    return paths

def estimate_performance(mdp, policy=None, horizon=None, n_episodes=1, gamma=0.9):
    paths = collect_episodes(mdp, policy, horizon, n_episodes)
    
    J = 0.
    for p in paths:
        df = 1
        sum_r = 0.
        for r in p["rewards"]:
            sum_r += df * r
            df *= gamma
        J += sum_r
    return J / n_episodes

def discretization_2d(x, y, binx, biny):
    _, _, _, binid = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
    return binid
