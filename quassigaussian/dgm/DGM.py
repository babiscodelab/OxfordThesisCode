# CLASS DEFINITIONS FOR NEURAL NETWORKS USED IN DEEP GALERKIN METHOD

# %% import needed packages
import tensorflow as tf

initializer1 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                               mode="fan_avg",
                                                               distribution="uniform")


# %% LSTM-like layer used in DGM (see Figure 5.3 and set of equations on p. 45) - modification of Keras layer class

class LSTMLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="identity"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer; 
                                   one of: "tanh" (default), "relu" or "sigmoid"
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''

        # create an instance of a Layer object (call initialize function of superclass of LSTMLayer)
        super(LSTMLayer, self).__init__()

        # add properties for layer including activation functions used inside the layer  
        self.output_dim = output_dim
        self.input_dim = input_dim

        if trans1 == "tanh":
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid

        if trans2 == "tanh":
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu
        elif trans2 == "identity":
            self.trans2 = tf.identity

        initializer2 = initializer1
        ### define LSTM layer parameters (use Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = self.add_weight("Uz", shape=[self.input_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)

        self.Ug = self.add_weight("Ug", shape=[self.input_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)
        self.Ur = self.add_weight("Ur", shape=[self.input_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)
        self.Uh = self.add_weight("Uh", shape=[self.input_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)

        # w vectors (weighting vectors for output of previous layer)        
        self.Wz = self.add_weight("Wz", shape=[self.output_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)
        self.Wg = self.add_weight("Wg", shape=[self.output_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)
        self.Wr = self.add_weight("Wr", shape=[self.output_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)
        self.Wh = self.add_weight("Wh", shape=[self.output_dim, self.output_dim],
                                    initializer=initializer1, trainable=True)

        # bias vectors
        self.bz = self.add_weight("bz", shape=[1, self.output_dim], initializer=initializer1, trainable=True)
        self.bg = self.add_weight("bg", shape=[1, self.output_dim], initializer=initializer1, trainable=True)
        self.br = self.add_weight("br", shape=[1, self.output_dim], initializer=initializer1, trainable=True)
        self.bh = self.add_weight("bh", shape=[1, self.output_dim], initializer=initializer1, trainable=True)

    # main function to be called 
    def call(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X .    

        Args:            
            S: output of previous layer
            X: data input
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''

        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))

        H = self.trans1(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))

        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))

        return S_new


# %% Fully connected (dense) layer - modification of Keras layer class

class DenseLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''

        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_weight("W", shape=[self.input_dim, self.output_dim],
                                  initializer=initializer1, trainable=True)

        # bias vectors
        self.b = self.add_weight("b", shape=[1, self.output_dim], initializer=initializer1,
                                 trainable=True)

        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation

    # main function to be called 
    def call(self, X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''

        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)

        if self.transformation:
            S = self.transformation(S)

        return S


# %% Neural network architecture used in DGM - modification of Keras Model class

class DGMNet(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''

        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet, self).__init__()

        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim + 1, transformation="tanh")

        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim + 1))

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation="relu")
        self.flatten_layer = tf.keras.layers.Flatten()
    # main function to be called


    def call(self, input):
        '''            `
        Args:
            t: sampled time inputs 
            x: sampled x space inputs
            y: sampled y space inputs
        Run the DGM model and obtain fitted function value at the inputs (t,x, y)
        '''

        # define input vector as time-space pairs
        (t, x, y) = input
        y = self.flatten_layer.call(y)
        #x = self.flatten_layer.call(x)
        X = tf.concat([t, x, y], 1)
        # call initial layer
        S = self.initial_layer.call(X)

        # call intermediate LSTM layers
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S, X)

        # call final LSTM layers
        result = self.final_layer.call(S)

        return result