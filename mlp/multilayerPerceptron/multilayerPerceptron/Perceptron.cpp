//
//  Perceptron.cpp
//  multilayerPerceptron
//
//  Created by Brian Desnoyers on 2/22/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include "Perceptron.hpp"
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

/**
 The sigmoud function. Nothing fancy here.

 @param value the input to the sigmoid function
 @return the result of the sigmoid function
 */
double sigmoid(double value) {
  return 1 / (1 + exp(-1 * value));
}

/**
 The derivative of the sigmoid function.

 @param sigmoidX the result of the sigmoid function
 @return the result of the function
 */
double derSigmoid(double sigmoidX) {
  return sigmoidX * (1 - sigmoidX);
}

/**
 Returns a real distribution for sampling weights. 
 This is based on the slide 94 from lecture 4 where we sample from [-b to b] 
 where b = sqrt(6) / sqrt(H_k + H_{k-1}). 
 Here b is multiplied by 4 for the sigmoid activation function.
 
 @param inputNodes the number of nodes in the starting layer for the weight
 @param outputNodes the number of nodes in the ending layer for the weight
 @return the distribution for random sampling
 */
uniform_real_distribution<double> distForWeights(int inputNodes, int outputNodes) {
  double upperBound = 4.0 * sqrt(6.0 / (inputNodes + outputNodes));
  double lowerBound = -1 * upperBound;
  
  return uniform_real_distribution<double>(lowerBound, upperBound);
}

SHLMLP::SHLMLP(int inputNodes, int outputNodes, int hiddenNodes, double learningRate, double momentum, double leastMeanSquareError) {
  this->inputNodes = inputNodes;
  this->outputNodes = outputNodes;
  this->hiddenNodes = hiddenNodes;
  
  this->learningRate = learningRate;
  this->leastMeanSquareError = leastMeanSquareError;
  this->momentum = momentum;
  
  // Set-up random number generator for generating weights
  random_device seedGenerator;
  mt19937_64 mersenneTwisterGenerator{seedGenerator()};
  
  // Initialize weights from inputs to hidden layer
  uniform_real_distribution<double> inpToHidDist = distForWeights(this->inputNodes, this->hiddenNodes);
  for (int imp = 0; imp < inputNodes; imp++) {
    vector<double> weightsFromImpI;
    for (int hid = 0; hid < hiddenNodes; hid++) {
      weightsFromImpI.push_back(inpToHidDist(mersenneTwisterGenerator));
    }
    this->inputToHiddenWeights.push_back(weightsFromImpI);
  }
  
  // Initialize weights from hidden to output layer
  uniform_real_distribution<double> hidToOutDist = distForWeights(this->hiddenNodes, this->outputNodes);
  for (int hid = 0; hid < hiddenNodes; hid++) {
    vector<double> weightsFromHidI;
    for (int out = 0; out < outputNodes; out++) {
      weightsFromHidI.push_back(hidToOutDist(mersenneTwisterGenerator));
    }
    this->hiddenToOutputWeights.push_back(weightsFromHidI);
  }
  
  // Resize vectors for holding node values
  // Note that bias vectors are not initalized (0)
  this->inputNodeValues.resize(this->inputNodes);
  this->hiddenNodeValues.resize(this->hiddenNodes);
  this->hiddenLayerBias.resize(this->hiddenNodes);
  this->outputNodeValues.resize(this->outputNodes);
  this->outputLayerBias.resize(this->outputNodes);
}

void SHLMLP::updateNodeValues() {
  // Calculate hidden layer node values
  for (int hiddenNodeIndex = 0; hiddenNodeIndex < this->hiddenNodes; hiddenNodeIndex++) {
    this->hiddenNodeValues[hiddenNodeIndex] = 0;
    for (int inputNodeIndex = 0; inputNodeIndex < this->inputNodes; inputNodeIndex++) {
      this->hiddenNodeValues[hiddenNodeIndex] += this->inputNodeValues[inputNodeIndex] * this->inputToHiddenWeights[inputNodeIndex][hiddenNodeIndex];
    }
    
    // Add bias
    this->hiddenNodeValues[hiddenNodeIndex] += hiddenLayerBias[hiddenNodeIndex];
    
    // Perform activation function (sigmoid)
    this->hiddenNodeValues[hiddenNodeIndex] = sigmoid(this->hiddenNodeValues[hiddenNodeIndex]);
  }
  
  // Calculate output node values
  for (int outputNodeIndex = 0; outputNodeIndex < this->outputNodes; outputNodeIndex++) {
    this->outputNodeValues[outputNodeIndex] = 0;
    for (int hiddenNodeIndex = 0; hiddenNodeIndex < this->hiddenNodes; hiddenNodeIndex++) {
      this->outputNodeValues[outputNodeIndex] += this->hiddenNodeValues[hiddenNodeIndex] * this->hiddenToOutputWeights[hiddenNodeIndex][outputNodeIndex];
    }
    
    // Add bias
    this->outputNodeValues[outputNodeIndex] += outputLayerBias[outputNodeIndex];
    
    // Perform activation function (sigmoid)
    this->outputNodeValues[outputNodeIndex] = sigmoid(this->outputNodeValues[outputNodeIndex]);
  }
}

void SHLMLP::train(vector<vector<int>> x, vector<int> y) {
  assert(x.size() == y.size()); // Ensure input vectors are of same size
  
  double meanSquareError;
  int currentEpoch = 1;
  double errorSquared = 0.0;
  double outputLayerDelta[this->outputNodes];
  double hiddenLayerDelta[this->hiddenNodes];
  
  vector<vector<double>> tempIHWeights = this->inputToHiddenWeights;
  vector<vector<double>> tempHOWeights = this->hiddenToOutputWeights;
  vector<vector<double>> prevIHWeights = this->inputToHiddenWeights;
  vector<vector<double>> prevHOWeights = this->hiddenToOutputWeights;
  
  do {
    // Reset the mean square error
    meanSquareError = 0.0;
    
    // Iterate across samples
    for (int s = 0; s < x.size(); s++) {
      // Populate input node values from sample s, and then forward propagate
      this->inputNodeValues = x[s];
      this->updateNodeValues();
      
      // Back-propagation
      for (int i = 0; i < this->outputNodes; i++) {
        double expectedP;
        if (i == y[s]) { // "correct node" representing our class- want to be 1
          expectedP = 1;
        } else {         // incorrect node- want to be 0
          expectedP = 0;
        }
        double error = expectedP - this->outputNodeValues[i];
        outputLayerDelta[i] = error * derSigmoid(this->outputNodeValues[i]);
        errorSquared += pow(error, 2);
      }
      
      // Find delta of hidden layer
      for (int hid = 0; hid < hiddenNodes; hid++) {
        hiddenLayerDelta[hid] = 0;
        for (int out = 0; out < outputNodes; out++) {
          // Our delta is proportional to the "blame" for the result in the output layer
          hiddenLayerDelta[hid] += outputLayerDelta[out] * hiddenToOutputWeights[hid][out];
        }
      }
      
      // Weight update step
      tempIHWeights = this->inputToHiddenWeights;
      tempHOWeights = this->hiddenToOutputWeights;
      
      // Multiply by derivative- per chain rule
      for (int hid = 0; hid < hiddenNodes; hid++) {
        hiddenLayerDelta[hid] *= derSigmoid(hiddenNodeValues[hid]);
      }
      
      // Update the input layer -> hidden layer weights
      for (int hid = 0; hid < this->hiddenNodes; hid++) {
        for (int inp = 0; inp < this->inputNodes; inp++) {
          this->inputToHiddenWeights[inp][hid] += this->momentum * (this->inputToHiddenWeights[inp][hid] - prevIHWeights[inp][hid]) + this->learningRate * hiddenLayerDelta[hid] * this->inputNodeValues[inp];
        }
        // Update the hidden layer bias
        // Conceptually, bias is a node that always outputs 1.
        // (We won't use the momentum term to update bias here.)
        hiddenLayerBias[hid] += this->learningRate * hiddenLayerDelta[hid] * 1;
      }
      
      // Update the hidden layer -> output layer weights
      for (int out = 0; out < outputNodes; out++) {
        for (int hid = 0; hid < hiddenNodes; hid++) {
          this->hiddenToOutputWeights[hid][out] += this->momentum * (this->hiddenToOutputWeights[hid][out] - prevHOWeights[hid][out]) + this->learningRate * outputLayerDelta[out] * this->hiddenNodeValues[hid];
        }
        // Update the output layer bias
        // Conceptually, bias is a node that always outputs 1.
        // (We won't use the momentum term to update bias here.)
        outputLayerBias[out] += this->learningRate * outputLayerDelta[out] * 1;
      }
      
      prevIHWeights = tempIHWeights;
      prevHOWeights = tempHOWeights;
      
      meanSquareError += errorSquared / (outputNodes + 1);
      errorSquared = 0;
    }
    cout << "Epoch " << currentEpoch << ": MSE = " << meanSquareError << endl;
    currentEpoch++; // increment epoch
  } while (meanSquareError >= (this->leastMeanSquareError + 0.0001));
}

int SHLMLP::test(vector<int> x) {
  this->inputNodeValues = x;
  this->updateNodeValues();
  
  double maxValue = -1;
  int maxIndex = -1;
  
  for (int i = 0; i < this->outputNodes; i++) {
    double value = this->outputNodeValues[i];
    if (maxIndex < 0 || value > maxValue) {
      maxValue = value;
      maxIndex = i;
    }
  }
  
  return maxIndex;
}
