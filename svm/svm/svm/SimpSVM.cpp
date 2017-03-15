//
//  SimpSVM.cpp
//  svm
//
//  Created by Brian Desnoyers on 2/18/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include "SimpSVM.hpp"
#include <assert.h>
#include <random>
#include <math.h>
#include <iostream>

// An alpha value has been updated, if it's deviation is at least this value.
const double ALPHA_CHANGE_DEVIATION = 0.00000003;

double dotProduct(vector<int> v1, vector<int> v2) {
  assert(v1.size() == v2.size());
  
  double result = 0;
  for (int i = 0; i < v1.size(); i++) {
    result += (v1[i] * v2[i]);
  }
  
  return result;
}

int sign(double d) {
  if (d > 0) {
    return 1;
  } else if (d < 0) {
    return -1;
  } else {
    return 0;
  }
}

//double dotProduct(vector<double> v1, vector<double> v2) {
//  int p1 = 0, p2 = 0;
//  double result = 0;
//  
//  while (p1 < v1.size() && p2 < v2.size()) {
//    if (v1[p1] == v2[p2]) {
//      result += v1[p1] * v2[p2];
//      p1++;
//      p2++;
//    } else if (v1[p1] > v2[p2]) {
//      p2++;
//    } else {
//      p1++;
//    }
//  }
//  
//  return result;
//}

BinSVM::BinSVM(double C, double tolerance, double maxPasses) {
  this->C = C; // 1
  this->tol = tolerance; // 0.001
  this->maxPasses = maxPasses;
}

void BinSVM::train(vector<vector<int>> features, vector<int> labels) {
  // Ensure size of feature and label vectors are the same
  assert(features.size() == labels.size());
  size_t m = features.size();
  
  // Initialize alphas and b to 0, save training labels
  this->alphas.resize(m, 0.0);
  this->b = 0.0;
  this->y = labels;
  this->x = features;
  
  // Set-up random number generator
  random_device seedGenerator;
  mt19937_64 mersenneTwisterGenerator{seedGenerator()};
  uniform_int_distribution<> inputDist{0, (int)m - 1};
  
  // Calculate dot products between all features to speed up computation later
  cout << "Pre-calculating linear kernel results..." << endl;
  this->dp.resize(m, vector<long>(m , 0));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      this->dp[i][j] = dotProduct(features[i], features[j]);
    }
  }
  cout << "Pre-calculation complete!" << endl;
  
  int passesWithoutChangingAlphas = 0;
  while (passesWithoutChangingAlphas < this->maxPasses) {
    // Count of updates to alpha values in this pass
    int alphaUpdateCount = 0;
    
    for (int i = 0; i < m; i++) {
      // Calculate E_i = f(x^{(i)}) - y^{(i)}
      double fxi = 0;
      for (int idx = 0; idx < m; idx++) {
        fxi += this->alphas[idx] * labels[idx] * this->dp[idx][i];
      }
      fxi += this->b;
      double E_i = fxi - labels[i];
      
      if ((((labels[i] * E_i) < (-1 * this->tol)) && (alphas[i] < C)) || ((labels[i] * E_i > this->tol) && this->alphas[i] > 0)) {
        // Select a random j != i
        int j;
        do {
          j = inputDist(mersenneTwisterGenerator);
        } while (j == i);
        // Calculate E_j = f(x^{(j)}) - y^{(j)}
        double fxj = 0;
        for (int idx = 0; idx < m; idx++) {
          fxj += this->alphas[idx] * labels[idx] * this->dp[idx][j];
        }
        fxj += b;
        double E_j = fxj - labels[j];
        
        // Save old alphas
        double oldAlpha_i = this->alphas[i];
        double oldAlpha_j = this->alphas[j];
        
        // Compute L and H
        // L <= a_j <= H must hold for a_j to satisfy the constraint that
        // 0 <= a_j <= C
        double L = 0;
        double H = 0;
        if (labels[i] == labels[j]) { // y^{(i)} == y^{(j)}
          L = max(0.0, this->alphas[i] + this->alphas[j] - C);
          H = min(C, this->alphas[i] + this->alphas[j]);
        } else {
          L = max(0.0, this->alphas[j] - this->alphas[i]);
          H = min(C, C + this->alphas[j] - this->alphas[i]);
        }
        
        if (L == H) {
          continue;
        }
        
        double eta = 2 * this->dp[i][j] - this->dp[i][i] - this->dp[j][j];
        if (eta >= 0) {
          continue;
        }
        
        // Compute optimal alpha_j
        this->alphas[j] -= ((labels[j] * (E_i - E_j)) / eta);
        // Clip alpha_j between H and L
        this->alphas[j] = max(this->alphas[j], L);
        this->alphas[j] = min(this->alphas[j], H);
        
        if (fabs(this->alphas[j] - oldAlpha_j) < ALPHA_CHANGE_DEVIATION) {
          // alpha_j unchanged
          this->alphas[j] = oldAlpha_j;
          continue;
        }
        
        // alpha_j updated
        alphaUpdateCount++;
        
        // Find new alpha_i
        this->alphas[i] += labels[i] * labels[j] * (oldAlpha_j - this->alphas[j]);
        
        // Compute b1 and b2
        double b1 = b - E_i - labels[i] * (this->alphas[i] - oldAlpha_i) * this->dp[i][i] - labels[j] * (this->alphas[j] - oldAlpha_j) * this->dp[i][j];
        double b2 = b - E_j - labels[i] * (this->alphas[i] - oldAlpha_i) * this->dp[i][j] - labels[j] * (this->alphas[j] - oldAlpha_j) * this->dp[j][j];
        
        // Compute b
        // Note: if both conditions hold, the values will both be equal
        if (0 < this->alphas[i] && this->alphas[i] < C) {
          this->b = b1;
        } else if (0 < this->alphas[j] && this->alphas[j] < C) {
          this->b = b2;
        } else {
          this->b = (b1 + b2) / 2.0;
        }
      }
    }
    if (alphaUpdateCount <= 0) {
      passesWithoutChangingAlphas++; // iterated w/o changing
    } else {
      passesWithoutChangingAlphas = 0;
    }
  }
  cout << endl;
}

double BinSVM::predict(vector<int> x) {
  // Calculate f(x)
  double fxi = 0;
  for (int idx = 0; idx < this->alphas.size(); idx++) {
    fxi += this->alphas[idx] * this->y[idx] * dotProduct(this->x[idx], x);
  }
  fxi += this->b;
  return fxi;
}

int BinSVM::predictClass(vector<int> x) {
  return sign(this->predict(x));
}
