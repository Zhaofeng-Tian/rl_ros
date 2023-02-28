import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions,mem_path,mem_ctr):
        self.mem_size = max_size
        self.mem_cntr = mem_ctr
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.mem_path = mem_path

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones 
    
    def save_mem(self):
        np.save(self.mem_path+'s.npy', self.state_memory)
        np.save(self.mem_path+'a.npy', self.action_memory)
        np.save(self.mem_path+'r.npy', self.reward_memory)
        np.save(self.mem_path+'s_.npy', self.new_state_memory)
        np.save(self.mem_path+'d.npy', self.terminal_memory)
        np.save(self.mem_path+'ctr.npy', self.mem_cntr)

    def load_mem(self, mem_path):
        self.state_memory = np.load(mem_path+'s.npy')
        self.action_memory = np.load(mem_path+'a.npy')
        self.reward_memory = np.load(mem_path+'r.npy')
        self.new_state_memory = np.load(mem_path+'s_.npy')
        self.terminal_memory = np.load(mem_path+'d.npy')
        self.mem_cntr = int(np.load(mem_path+'ctr.npy'))

