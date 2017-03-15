//
//  Perceptron.hpp
//  Perceptron
//
//  Created by Brian Desnoyers on 2/23/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#ifndef Perceptron_hpp
#define Perceptron_hpp

#include <functional>
#include <stdio.h>
#include <vector>

using namespace std;

/**
 Returns the dot product between two equal lengthed vectors `v1` and `v2`
 
 @param v1 the 1st vector
 @param v2 the 2nd vector
 @return the dot product between two equal lengthed vectors `v1` and `v2`
 */
double dotProduct(vector<double> v1, vector<double> v2);

/**
 A polynomial kernel with p = 3.
 
 @param v1 the first vector
 @param v2 the second vector
 @return the result of the kernel function
 */
double polynomialKernel(vector<double> v1, vector<double> v2);

/**
 A gaussian kernel w/ sigma = 1.0 (lambda = 0.5).
 
 @param v1 the 1st vector
 @param v2 the 2nd vector
 @return the result of the kernel function
 */
double gaussianKernel(vector<double> v1, vector<double> v2);

/**
 A laplacian kernel w/ sigma = 1.0.
 
 @param v1 the 1st vector
 @param v2 the 2nd vector
 @return the result of the kernel function
 */
double laplacianKernel(vector<double> v1, vector<double> v2);

/**
 A single-node perceptron class using the primal form.
 */
class Perceptron {
private:
  // the weight vector
  vector<double> w;
  // the bias
  double b;
  
public:
  /**
   Trains the perceptron with features 'x' and labels 'y'.

   @param x the features to train
   @param y the corresponding labels
   */
  void train(vector<vector<double>> x, vector<int> y);
  
  /**
   Returns the weights after training.

   @return the weights after training.
   */
  vector<double> getWeights();
  
  /**
   Returns the normalized weights calculated by dividing by the bias (w_0).

   @return the normalized weights
   */
  vector<double> getNormalizedWeights();
  
  /**
   Returns the bias after training.

   @return the bias after training
   */
  double getBias();
};

/**
 A single-node perceptron class using the dual form (kernelized).
 */
class DualPerceptron {
private:
  // the vector of counts
  vector<double> m;
  // the weight vector- calculated after training
  vector<double> w;
  // the bias
  double b;
  // the kernel function
  function<double(vector<double>, vector<double>)> kernel;
  
public:
  
  /**
   Initializes a kernelized perceptron with the kernel `kernel`.
   
   @param kernel the kernel
   */
  DualPerceptron(function<double(vector<double>, vector<double>)> kernel);
  
  /**
   Trains the perceptron with features 'x' and labels 'y'.
   
   @param x the features to train
   @param y the corresponding labels
   */
  void train(vector<vector<double>> x, vector<int> y);
  
  /**
   Returns the weights after training.
   
   @return the weights after training.
   */
  vector<double> getWeights();
  
  
  /**
   Returns the normalized weights calculated by dividing by the bias (w_0).
   
   @return the normalized weights
   */
  vector<double> getNormalizedWeights();
  
  
  /**
   Returns the bias after training.
   
   @return the bias after training
   */
  double getBias();
};

#endif /* Perceptron_hpp */
