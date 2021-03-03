
class DgmParameters():

    def __init__(self, num_layers=3, nodes_per_layer=50, learning_rate=0.001, sampling_stages=20, steps_per_sample=5,
                 n_sim_interior=1000, n_sim_terminal=100):

        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.learning_rate = learning_rate
        self.sampling_stages = sampling_stages
        self.steps_per_sample = steps_per_sample
        self.n_sim_interior = n_sim_interior
        self.n_sim_terminal = n_sim_terminal