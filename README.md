Multi-level Optimization for Neural Network

Workflow:
1. define an old tf.Graph
2. train the network with data with [train_load_net] function
3. to expand a network, call [extract_weight] function, weights will be returned as numpy.ndarray
4. call [Net2Net] functions to expand weights and layers
5. define new tf.Graph
6. call [train_load_net] to restart from a checkpoint with new graph and new weights
