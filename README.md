# Description
Implementation of feed-forward neural network with customizable number of layers and number of nodes in each layer. Trains by gradient descent with backpropagation. Saves weights when training for later use.

# Running the network
Place configuration files in the input folder and specify which one to use in launch.json
The following formats of files are used for the three modes of operation.

Training from scratch:
```
Training
Num layers = 4
Num activations in layer 0 = 2
Num activations in layer 1 = 2
Num activations in layer 2 = 3
Num activations in layer 3 = 3

lambda = 0.1
errorThreshold = 0.001
Nmax = 1000000
outputFileName = outputOftrainXORWith2-2-3-3.txt
saveWeightsInterval = 40000

isRandomlyInitialized = true
lower bound for random weights = -1.0
upper bound for random weights = 1.5

numTestCases = 4
0.0 0.0 
0.0 0.0 0.0 

0.0 1.0 
0.0 1.0 1.0 

1.0 0.0 
0.0 1.0 1.0 

1.0 1.0 
1.0 1.0 0.0 
```

Training from given weights:
```
Training
Num layers = 3
Num activations in layer 0 = 2
Num activations in layer 1 = 5
Num activations in layer 2 = 3

lambda = 0.1
errorThreshold = 0.001
Nmax = 1000000
outputFileName = outputOfTrainXORWith5AndGoodStartingWeights.txt
saveWeightsInterval = 40000

isRandomlyInitialized = false
weights = weights.txt

numTestCases = 4
0.0 0.0 
0.0 0.0 0.0 

0.0 1.0 
0.0 1.0 1.0 

1.0 0.0 
0.0 1.0 1.0 

1.0 1.0 
1.0 1.0 0.0 
```

Running:
```
Running
Num layers = 3
Num activations in layer 0 = 2
Num activations in layer 1 = 5
Num activations in layer 2 = 3

outputFileName = output_run_example.txt
fileNameWeights = weights_outputOfTrainXORWith5AndGoodStartingWeights.txt

Input
1.0 1.0 
```