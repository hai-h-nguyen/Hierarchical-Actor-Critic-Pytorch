import numpy as np

class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size, is_top_layer=False):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size
        self.is_top_layer = is_top_layer

    def add(self, experience):
        if self.is_top_layer:
            assert len(experience) == 6, 'Experience must be of form (s, a, r, s, t, grip_info\')'
            assert type(experience[4]) == bool
        else:
            assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
            assert type(experience[5]) == bool

        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer.  Only removing a single transition slows down performance
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self):
        if self.is_top_layer:
            states, actions, rewards, new_states, is_terminals = [], [], [], [], []
        else:
            states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=self.batch_size)
        
        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])

            if self.is_top_layer:
                is_terminals.append(self.experiences[i][4])
            else:
                goals.append(self.experiences[i][4])
                is_terminals.append(self.experiences[i][5])                

        if self.is_top_layer:
            return states, actions, rewards, new_states, is_terminals
        else:
            return states, actions, rewards, new_states, goals, is_terminals