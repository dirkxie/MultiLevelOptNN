# Multi-level Optimization for Neural Network

## Workflow:
1. define an old tf.Graph <br />
2. train the network with data with [train_load_net] function <br />
3. to expand a network, call [extract_weight] function, weights will be returned as numpy.ndarray <br />
4. call [Net2Net] functions to expand weights and layers <br />
5. define new tf.Graph <br />
6. call [train_load_net] to restart from a checkpoint with new graph and new weights <br />

## To be finished [LungNodule]:
1. expand conv2 twice as wide <br />
2. add 1 more convolutional layer after conv3 <br />
3. add skip layer (cautious) <br />

## Notice when using K40C GPU
1. Default session will only allocate 512MB of GDDR. Add the following to your session: <br />
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= any 0~1 number) <br />
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess: <br />

2. Total GDDR of K40C is 10GB, call nvidia-smi to see memory allocation status first before run. <br />
   
3. Let us share memory by calculate how much memory is needed first and then change the factor in code, <br />
