
import os, sys
sys.path.append( os.path.join(os.getcwd(), '..') )

from utils import *

''' change wanted agent_fn import before running '''
# from atari.DQN import agent_fn
from atari.IQN import agent_fn


def run_agent(GAME, name, run_number):
    env = gym.make(GAME)
    S, A = env.observation_space.shape[0], env.action_space.n
    agent = agent_fn(S,A)
    exp = Experiment(env, agent)
    exp.agent_state_preprocessing_fn = lambda s: s[None,:] / 256.
    df = exp.run(5*10, exp.train_steps//10, name=name, run_number=run_number)
    return df

if __name__ == '__main__':
    GAME = 'Pong-ram-v4'
    name = 'IQN'
    num_runs = os.cpu_count()
    args = [(GAME, name, i) for i in range(num_runs)]
    pool = mp.Pool()
    dfs = pool.starmap(run_agent, args)
    df = pd.DataFrame({})
    for d in dfs:
        df = df.append(d)
    os.chdir('..')
    save_my_benchmark(df, GAME, name)
    