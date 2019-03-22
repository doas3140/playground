from utils import *

GAME = 'Pong-ram-v0'
env_fn = lambda: gym.make(GAME)
env = env_fn()
S, A = env.observation_space.shape[0], env.action_space.n

# Model

class Model(tf.keras.Model):
    def __init__(self, S, A):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(50, 'relu')
        self.W2 = tf.keras.layers.Dense(50, 'relu')
        self.V = tf.keras.layers.Dense(1)
        self.P = tf.keras.layers.Dense(A, 'softmax')
        self.build(input_shape=(None,S))
    
    def call(self, x):
        x = self.W1(x)
        x = self.W2(x)
        return self.P(x), self.V(x)
    
create_nn = lambda S=S,A=A: Model(S,A)

# Walker Agent (walks and collects experience)

class WalkerAgent():
    def __init__(self, num_states, num_actions, weights, gamma=0.99):
        self.nn = create_nn(num_states, num_actions)
        self.nn.set_weights(weights)
        self.y = gamma
        
    def first_step(self, s, sdict): # [1,S]
        pi, v = self.nn(s) # [1,A], []
        a = tf.random.categorical(tf.math.log(pi), num_samples=1) # [1,1]
        a = int(a[0,0]) # []
        self.E = []
        self.s, self.a, self.pi = s, a, pi # save prev state + action
        return a
        
    def step(self, r, s, sdict): # [], [1,S]
        # append tuple to experience replay
        self.E.append([self.s, self.a, self.pi, r, s]) # s, a, pi, r, s'
        pi, v = self.nn(s) # [1,A], []
        a = tf.random.categorical(tf.math.log(pi), num_samples=1) # [1,1]
        a = int(a[0,0]) # []
        self.s, self.a, self.pi = s, a, pi # save previous state, action
        return a
    
    def last_step(self, r, sdict):
        self.E.append([self.s, self.a, self.pi, r, None]) # s, a, pi, r, s'
        # prerprocess reward
        R = np_map(lambda x: x[3], self.E) # [e]
        G = discount_rewards(R, self.y)[:,None] # [e,1]
        G = (G + 21) / 21*2
        for i,_ in enumerate(self.E): self.E[i][3] = G[i]
        
walker_fn = lambda S,A,weights: WalkerAgent(S, A, weights, gamma=0.99)

def run_walker(num_states, num_actions, env, model_weights):
    agent = walker_fn(num_states, num_actions, model_weights)
    exp = Experiment(env, agent, use_tqdm=False)
    # when getting state from ram return s w/ shape [1,s] divided by 256
    exp.agent_state_preprocessing_fn = lambda s: s[None,:] / 256
    _ = exp.run(1,1) # run a single experiment with no max length
    return exp.agent.E # experience

# gradient calculation using experience (D)

_min, clip, log = tf.minimum, tf.clip_by_value, tf.math.log
def ppo_policy(A_new, A_old, Adv, A, e=0.2): # [b,A], [b,1], [b,A], []
    # pi, pi_old, Advantage=G-V, Action that was taken (one-hot), epsilon
    r = A_new/A_old # [e,A]
    return - A * _min( r*Adv, clip(r,1-e,1+e)*Adv )

def calc_grads(nn, D, batch_size=1024, num_actions=A):
    B = random.sample(D, batch_size)
    # need squeeze, because self.s/a is [1,S/A], so S=[e,1,S/A] -> [e,S/A]
    S = np.squeeze(np_map(lambda x: x[0], B)) # [e,S], where e - number steps in episode
    if S.ndim == 1: S = S[None,:] # for cases if example size is 1 and squeezed
    A = np.squeeze(np_map(lambda x: x[1], B)) # [e]
    A = one_hot(A.astype(int), C=num_actions) # [e,A]
    A_old = np.squeeze(np_map(lambda x: x[2], B)) # [e,A]
    A_old = tf.clip_by_value(A_old, 1e-6, 1-1e-6) # [e,A]
    G = np.squeeze(np_map(lambda x: x[3], B))[:,None] # [e,1]
    # calc loss
    with tf.GradientTape() as tape:
        A_hat, V_hat = nn(S) # [e,A], [e,1]
        A_hat = tf.clip_by_value(A_hat, 1e-6, 1-1e-6) # clip tf.log(0)
        Adv = tf.stop_gradient(G - V_hat) # [e,A] (advantage)
        policy_loss = ppo_policy(A_hat, A_old, Adv, A)
        entropy = - A_hat * tf.math.log(A_hat)
        value_loss = (G - V_hat)**2 # [e,A]
        each_loss = policy_loss + 0.01*entropy + 0.5*value_loss # [e,A]
        loss = tf.reduce_mean(tf.reduce_sum(each_loss, axis=-1)) # avg of all examples
    # calc + return gradients
    grads = tape.gradient(loss, nn.trainable_variables)
    return grads

# Master Agent

class MasterAgent():
    def __init__(self, num_states, num_actions, env_fn, lr, num_cores=os.cpu_count()):
        self.S, self.A = num_states, num_actions
        self.env_fn = env_fn
        self.num_cores = num_cores
        self.nn = create_nn(num_states,num_actions)
        self.optimizer = tf.optimizers.Adam(lr)
        self.pool = mp.Pool(num_cores) # used to distribute work
        self.D = deque(maxlen=num_cores*10000)
        self.t = -2000
        
    def first_step(self, s, sdict): # [1,S]
        a = np.random.randint(self.A) # [] (random move)
        # init stats
        sdict['entropy'] += 0
        return a
        
    def step(self, r, s, sdict): # [], [1,S]
        pi, v = self.nn(s) # [1,A]
        a = tf.random.categorical(tf.math.log(pi), num_samples=1) # [1,1]
        a = int(a[0,0])
        # every n step modify nn w/ gradients respect to experience(self.D)
        if self.t > 500:
            self.t = 0
            grads = calc_grads(self.nn, self.D)
            self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        self.t += 1
        # save stats
        sdict['entropy'] += - np.sum(pi[0] * np.log(pi[0]))
        return a # []
    
    def last_step(self, r, sdict):
        # run walkers
        args = [(self.S, self.A, self.env_fn(), self.nn.get_weights()) for i in range(self.num_cores)]
        walkers_experience = self.pool.starmap(run_walker, args)
        for E in walkers_experience: self.D.extend(E)
        

agent_fn = lambda S,A,env_fn: MasterAgent(S, A, env_fn, lr=1e-4, num_cores=os.cpu_count())


if __name__ == '__main__':
    env = env_fn()
    exp = Experiment(env, agent_fn(S,A,env_fn))
    try: exp.agent.nn.load_weights('./weights/pong')
    except: pass
    exp.agent_state_preprocessing_fn = lambda s: s[None,:] / 256
    exp.iter = 0
    exp.show_play(100)
    # df = exp.run(2000, 1, name='')
    # try: df = df.append(pd.read_pickle('./weights/pong'))
    # except: pass
    # df.to_pickle('./weights/pong')
    # exp.agent.nn.save_weights('./weights/pong')

