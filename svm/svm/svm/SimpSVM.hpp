//
//  SimpSVM.hpp
//  svm
//
//  Created by Brian Desnoyers on 2/18/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#ifndef SimpSVM_hpp
#define SimpSVM_hpp

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

/**
 Finds the dot product of two equal length (1D) vectors v1 and v2.

 @param v1 the first vector
 @param v2 the second vector
 @return the dot product
 */
double dotProduct(vector<double> v1, vector<double> v2);

class BinSVM {
private:
  // Input parameters
  double C; // regularization parameter
  double tol; // numerical tolerance
  int maxPasses; // max # of times to iterate over alphas w/o changing
  
  // Solution
  vector<double> alphas; // A vector for holding the Lagrange multipliers for solution
  double b; // The threshold for solution
  
  // Caches
  vector<int> y; // A vector containing the training labels
  vector<vector<int>> x; // A vector containing the training features
  vector<vector<long>> dp; // The cached dot products between all features
  
  /**
   Predicts the prediction without the sign operator applied.
   This is called internally by `predictClass`.
   
   @param x the feature vector
   @return the prediction
   */
  double predict(vector<int> x);
  
public:
  /**
   Constructor for a new SVM classifier.
   
   @param C           the regularization parameter
   @param tolerance   the tolerance parameter
   @param maxPasses   the number of passes w/o changing alphas- terminating cond.
   */
  BinSVM(double C, double tolerance, double maxPasses);
  
  /**
   Trains the model on the feature vectors `features` and their corresponding
   `labels`.

   @param features the feature vectors to train
   @param labels the corresponding labels
   */
  void train(vector<vector<int>> features, vector<int> labels);
  
  /**
   Predicts the class associated with feature vector `x`.

   @param x the feature vector
   @return the predicted class
   */
  int predictClass(vector<int> x);
};


#endif /* SimpSVM_hpp */
