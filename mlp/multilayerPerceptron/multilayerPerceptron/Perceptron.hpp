//
//  Perceptron.hpp
//  multilayerPerceptron
//
//  Created by Brian Desnoyers on 2/22/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#ifndef Perceptron_hpp
#define Perceptron_hpp

#include <stdio.h>
#include <vector>
#include <random>

using namespace std;

// A single hidden layer multilayer perceptron
class SHLMLP {
private:
  // The current input node values- used for propagation
  vector<int> inputNodeValues;
  // The current hidden node values
  vector<double> hiddenNodeValues;
  // The current output node values
  vector<double> outputNodeValues;
  // 2D vector of weights between each node in the input layer to the hidden layer
  vector<vector<double>> inputToHiddenWeights;
  // 2D vector of weights between each node in the hidden layer to the output layer
  vector<vector<double>> hiddenToOutputWeights;
  // The bias vector for the hidden layer
  vector<double> hiddenLayerBias;
  // The bias vector for the output layer
  vector<double> outputLayerBias;
  
  int inputNodes;   // the number of input nodes
  int outputNodes;  // the number of output nodes
  int hiddenNodes;  // the number of hidden nodes
  
  double learningRate; // the learning rate (gamma)
  double leastMeanSquareError; // the stopping condition- when > MSE
  double momentum; // the gradient descent momentum term
  
public:
  /**
   Constructs a single hidden layer multilayer perceptron.
   
   @param inputNodes              the feature count
   @param outputNodes             the number of output classes
   @param hiddenNodes             the number of hidden nodes
   @param learningRate            the learning rate
   @param momentum                the momentum
   @param leastMeanSquareError    the least mean square error- terminating condition
   */
  SHLMLP(int inputNodes, int outputNodes, int hiddenNodes, double learningRate, double momentum, double leastMeanSquareError);
  
  /**
   Updates the node values through propagation. 
   */
  void updateNodeValues();
  
  /**
   Trains the network with features `x` and their corresponding labels `y`.

   @param x the training features
   @param y the labels- should contain values from [0, outputNodes)
   */
  void train(vector<vector<int>> x, vector<int> y);
  
  /**
   Predicts the class for features 'x'.

   @param x the features to predict
   @return the predicted class
   */
  int test(vector<int> x);
  
};

#endif /* Perceptron_hpp */
