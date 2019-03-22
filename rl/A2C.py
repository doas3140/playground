from utils import *

# Model

class Model(tf.keras.Model):
    def __init__(self, S, A):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(50, 'relu', input_dim=S)
        self.W2 = tf.keras.layers.Dense(50, 'relu', input_dim=50)
        self.V = tf.keras.layers.Dense(1, input_dim=50)
        self.P = tf.keras.layers.Dense(A, 'softmax', input_dim=50)
        self.build(input_shape=(None,S))
    
    def call(self, x):
        x = self.W1(x)
        x = self.W2(x)
        return self.P(x), self.V(x)
    
create_nn = lambda S,A: Model(S,A)

# Worker Agent

class WorkerAgent():
    def __init__(self, num_states, num_actions, model_weights, gamma=0.99):
        self.S, self.A, self.y = num_states, num_actions, gamma
        self.global_nn = create_nn(num_states,num_actions)
        self.global_nn.set_weights(model_weights)
        self.optimizer = tf.optimizers.Adam(1e-2)
        
    def first_step(self, s, sdict): # [1,S]
        a = np.random.randint(self.A) # [] (random move)
        self.E = []
        self.s, self.a = s, a # save prev state + action
        return a
        
    def step(self, r, s, sdict): # [], [1,S]
        # append tuple to experience replay
        self.E.append([self.s, self.a, r, s, False]) # s, a, r, s', done(bool)
        pi, v = self.global_nn(s) # [1,A]
        a = tf.random.categorical(tf.math.log(pi), num_samples=1) # [1,1]
        a = int(a[0,0]) # []
        self.s, self.a = s, a # save previous state, action
        return a
    
    def last_step(self, r, sdict):
        self.E.append([self.s, self.a, r, None, True]) # s, a, r, s', done
        # need squeeze, because self.s/a is [1,S/A], so S=[e,1,S/A] -> [e,S/A]
        S = np.squeeze(np_map(lambda x: x[0], self.E)) # [e,S], where e - number steps in episode
        if S.ndim == 1: S = S[None,:] # for cases if example size is 1 and squeezed
        A = np.squeeze(np_map(lambda x: x[1], self.E)) # [e]
        A = one_hot(A.astype(int), C=self.A) # [e,A]
        # prerprocess reward
        R = np_map(lambda x: x[2], self.E) # [e]
        G = discount_rewards(R, self.y, normalize=True)[:,None] # [e,1]
        # train nn for 1 epoch
        with tf.GradientTape() as tape:
            A_hat, V_hat = self.global_nn(S) # [e,A]
            A_hat = tf.clip_by_value(A_hat, 1e-6, 1-1e-6) # clip tf.log(0)
            Adv = G - V_hat # [e,A] (advantage)
            policy_loss = -A * tf.math.log(A_hat) * tf.stop_gradient(Adv) # [e,A]
            entropy = A_hat * tf.math.log(A_hat) # [e,A]
            value_loss = Adv**2 # [e,A]
            each_loss = policy_loss - 0.01*entropy + value_loss # [e,A]
            loss = tf.reduce_mean(tf.reduce_sum(each_loss, axis=-1)) # avg of all examples
        # calc gradients and save them
        self.grads = tape.gradient(loss, self.global_nn.trainable_variables)
        # self.grads, _ = tf.clip_by_global_norm(self.grads, 40.0) # clip each gradient
        # self.optimizer.apply_gradients(zip(self.grads, self.global_nn.trainable_variables)) # no need to apply, we just save them


worker_fn = lambda S,A,model_weights: WorkerAgent(S, A, model_weights, gamma=0.99)

def run_worker(num_states, num_actions, env, weights):
    agent = worker_fn(num_states, num_actions, weights)
    exp = Experiment(env, agent, use_tqdm=False)
    _ = exp.run(1,1) # run a single experiment with no max length
    return exp.agent.grads

# Master Agent

class MasterAgent():
    def __init__(self, num_states, num_actions, env_fn, lr, num_cores=os.cpu_count()):
        self.S, self.A = num_states, num_actions
        self.env_fn = env_fn
        self.num_cores = num_cores
        self.nn = create_nn(num_states,num_actions)
        self.optimizer = tf.optimizers.Adam(lr)
        self.pool = mp.Pool(num_cores) # used to distribute work
        
    def first_step(self, s, sdict): # [1,S]
        a = np.random.randint(self.A) # [] (random move)
        return a
        
    def step(self, r, s, sdict): # [], [1,S]
        pi, v = self.nn(s) # [1,A]
        a = tf.random.categorical(tf.math.log(pi), num_samples=1) # [1,1]
        return int(a[0,0]) # []
    
    def last_step(self, r, sdict):
        args = [(self.S, self.A, self.env_fn(), self.nn.get_weights()) for i in range(self.num_cores)]
        workers_grads = self.pool.starmap(run_worker, args)
        for grads in workers_grads:
            self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        

agent_fn = lambda S,A,env_fn: MasterAgent(S, A, env_fn, lr=1e-4, num_cores=os.cpu_count())


if __name__ == '__main__':
    GAME = 'Pong-ram-v0'
    env_fn = lambda: gym.make(GAME)
    env = env_fn()
    S, A = env.observation_space.shape[0], env.action_space.n
    exp = Experiment(env, agent_fn(S,A,env_fn))
    exp.agent_state_preprocessing_fn = lambda s: s[None,:] / 256
    df = exp.run(100, 1, name='')

