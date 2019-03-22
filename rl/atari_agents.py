
from utils import *

def run_agent(GAME, name, run_number):
    env = gym.make(GAME)
    S, A = env.observation_space.shape[0], env.action_space.n
    agent = agent_fn(S,A)
    exp = Experiment(env, agent)
    exp.agent_state_preprocessing_fn = lambda s: s[None,:] / 256.
    df = exp.run(1000, 1, name=name, run_number=run_number)
    return df

class Model(tf.keras.Model):
    def __init__(self, S, A):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(128, 'relu')
        self.W2 = tf.keras.layers.Dense(128, 'relu')
        self.W3 = tf.keras.layers.Dense(A)
        self.build((None,S))
    
    def call(self, x):
        x = self.W1(x)
        x = self.W2(x)
        return self.W3(x)
    
create_nn = lambda S,A: Model(S,A)

class DQNAgent():
    def __init__(self, num_states, num_actions, lr=1e-4, gamma=0.99):
        self.S, self.A = num_states, num_actions
        self.optimizer = tf.optimizers.Adam(lr)
        self.y = gamma
        self.Qnn = create_nn(self.S, self.A) # Q online network
        self.Tnn = create_nn(self.S, self.A) # Target network
        self.Tnn.set_weights( self.Qnn.get_weights() )
        self.double_dqn = True
        self.batch_size = 64 # batch size
        self.swap_iters = 5000 # if iter > this: swap/copy networks
        self._step = 1 # step count to construct epsilon and swap networks
        self.D = deque(maxlen=100000) # experience replay buffer
        self.step2epsilon = create_epsilon_fn([
            # from_e, to_e, from_t, to_t (linear change)
            [1.0, 0.5, 0.0, 5e5],
            [0.5, 0.2, 5e5, 1e6],
            [0.2, 0.1, 1e6, 2e6],
            [0.1, 0.0, 2e6, 4e6]
        ])
        
    def first_step(self, s, sdict): # [1,S]
        a = np.random.randint(self.A) # [] (random move)
        self.s, self.a = s, a # save prev state + action + q value
        # init plot values if not inited
        for a in range(self.A):
            sdict[f'action{a}'] += 0
            sdict[f'Qaction{a}'] += 0
        sdict['epsilon'] += 0
        sdict['loss'] += 0
        return a
        
    def step(self, r, s, sdict): # [], [1,S]
        # append tuple to experience replay
        self.D.append([self.s, self.a, r, s, False]) # s, a, r, s', done(bool)
        if self._step % 2 == 0: # ecery second frame update loss
            if len(self.D) < self.batch_size: loss = 0
            else: loss = self.update_weights(self.batch_size, sdict)
            sdict['loss'] += float(loss)
        # copy Q network to T network
        if self._step % self.swap_iters == 0:
            self.Tnn.set_weights( self.Qnn.get_weights() )
        # update other vars
        epsilon = self.step2epsilon(self._step)
        sdict['epsilon'] += epsilon
        self._step += 1
        # select action from Q network
        q = self.Qnn(s) # [1,A]
        a = epsilon_greedy(q, epsilon) # []
        for a_ in range(self.A): sdict[f'Qaction{a_}'] += float(q[0,a_])
        sdict[f'action{a}'] += 1
        # remember last state + action
        self.s, self.a = s, a
        return a
    
    def update_weights(self, b, sdict): # batch size
        # select batch from experience replay
        B = random.sample(self.D, b) # random batch from experience replay
        # calculate Q values in s'
        S = np.squeeze(np_map(lambda x: x[3], B)) # [b,S]
        R = np.squeeze(np_map(lambda x: x[2], B)) # [b]
        # calculate new Q values in s when doing a action
        mask = ~np.squeeze(np_map(lambda x: x[4], B)).astype(bool) # [b] (mask if it is last action)
        Q = self.Tnn(S)
        Qa = R + self.y * np.max(Q, axis=-1) * mask # [b] (bellman error)
        # leave other action values the same in s
        S = np.squeeze(np_map(lambda x: x[0], B)) # [b,S]
        A = np.squeeze(np_map(lambda x: x[1], B)).astype(int) # [b]
        Q = self.Qnn(S).numpy() # [b,A]
        Q[np.arange(b),A] = Qa # [b,A]
        # train nn for 1 epoch
        with tf.GradientTape() as tape:
            Q_hat = self.Qnn(S) # [e,A]
            each_loss = tf.reduce_sum( (Q-Q_hat)**2 , axis=-1 ) # [e]
            loss = tf.reduce_mean(each_loss) # avg of all examples
        # calc + update gradients
        nn_vars = self.Qnn.trainable_variables
        grads = tape.gradient(loss, nn_vars)
        self.optimizer.apply_gradients(zip(grads, nn_vars))
        return loss
    
    def last_step(self, r, sdict):
        self.D.append([self.s, self.a, r, self.s, True]) # s, a, r, s', done

agent_fn = lambda S,A,env_fn=None: DQNAgent(num_states=S, num_actions=A)

if __name__ == '__main__':
    GAME = 'Pong-ram-v4'
    name = 'DQN'
    num_runs = os.cpu_count()
    args = [(GAME, name, i) for i in range(num_runs)]
    pool = mp.Pool()
    dfs = pool.starmap(run_agent, args)
    df = pd.DataFrame({})
    for d in dfs:
        df = df.append(d)
    save_my_benchmark(df, GAME, name)
    