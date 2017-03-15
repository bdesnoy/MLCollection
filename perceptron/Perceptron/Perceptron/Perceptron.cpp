//
//  Perceptron.cpp
//  Perceptron
//
//  Created by Brian Desnoyers on 2/23/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include "Perceptron.hpp"
#include <assert.h>
#include <iostream>
#include <math.h>

double dotProduct(vector<double> v1, vector<double> v2) {
  assert(v1.size() == v2.size());
  
  double result = 0;
  for (int i = 0; i < v1.size(); i++) {
    result += (v1[i] * v2[i]);
  }
  
  return result;
}

vector<double> subtract(vector<double> v1, vector<double> v2) {
  assert(v1.size() == v2.size());
  
  vector<double> result;
  result.resize(v1.size());
  for (int i = 0; i < result.size(); i++) {
    result[i] = v1[i] - v2[i];
  }
  return result;
}

double magnitude(vector<double> v1) {
  double result = 0;
  for (int i = 0; i < v1.size(); i++) {
    result += pow(v1[i], 2);
  }
  return sqrt(result);
}

double polynomialKernel(vector<double> v1, vector<double> v2) {
  double p = 3;
  return pow((1 + dotProduct(v1, v2)), p);
}

double gaussianKernel(vector<double> v1, vector<double> v2) {
  double sigma = 1;
  return exp((-1 * pow(magnitude(subtract(v1, v2)), 2)) / (2 * pow(sigma, 2)));
}

double laplacianKernel(vector<double> v1, vector<double> v2) {
  double sigma = 1;
  return exp((-1 * magnitude(subtract(v1, v2))) / sigma);
}

void Perceptron::train(vector<vector<double>> x, vector<int> y) {
  // Initialize weight and bias = 0
  if (!this->w.empty()) {
    this->w.erase(w.begin());
  }
  this->w.resize(x[0].size(), 0.0);
  this->b = 0;
  
  int mistakes;
  int iterationsUntilConvergence = 0;
  do {
    mistakes = 0;
    iterationsUntilConvergence++;
    for (int i = 0; i < x.size(); i++) {
      double yTest = dotProduct(this->w, x[i]) + this->b;
      // Check if prediction (sign(yTest)) matches label
      if (yTest * y[i] <= 0) { // mistake
        mistakes++;
        for (int j = 0; j < x[0].size(); j++) { // update weight
          w[j] += y[i] * x[i][j];
        }
        b += y[i];
      }
    }
    cout << "Iteration " << iterationsUntilConvergence << ": w/ " << mistakes << " mistakes" << endl;
  } while (mistakes != 0);
}

vector<double> Perceptron::getWeights() {
  return this->w;
}

vector<double> Perceptron::getNormalizedWeights() {
  vector<double> normWeights = this->w;
  for (int i = 0; i < normWeights.size(); i++) {
    normWeights[i] /= -this->b;
  }
  return normWeights;
}


double Perceptron::getBias() {
  return this->b;
}


DualPerceptron::DualPerceptron(function<double(vector<double>, vector<double>)> kernel) {
  this->kernel = kernel;
}

void DualPerceptron::train(vector<vector<double>> x, vector<int> y) {
  // Initialize m, w, and b
  if (!this->m.empty()) {
    this->m.erase(m.begin());
  }
  this->m.resize(x.size(), 0.0);
  if (!this->w.empty()) {
    this->w.erase(w.begin());
  }
  this->w.resize(x[0].size(), 0.0);
  this->b = 0;
  
  // Calculate result of kernals to speed up computation later
  double k[x.size()][x.size()];
  for (int i = 0; i < x.size(); i++) {
    for (int j = 0; j < x.size(); j++) {
      k[i][j] = this->kernel(x[i], x[j]);
    }
  }
  
  int mistakes;
  int iterationsUntilConvergence = 0;
  do {
    mistakes = 0;
    iterationsUntilConvergence++;
    for (int i = 0; i < x.size(); i++) {
      // training sample i
      double yTest = 0;
      for (int j = 0; j < x.size(); j++) {
        yTest += (k[j][i] * this->m[j] * y[j]);
      }
      yTest += this->b;
      // Check if prediction (sign(yTest)) matches label
      if (yTest * y[i] <= 0) { // mistake
        mistakes++;
        this->m[i]++; // increment m count
        b += y[i];
      }
    }
    cout << "Iteration " << iterationsUntilConvergence << ": w/ " << mistakes << " mistakes" << endl;
  } while (mistakes != 0);
  
  // Calculate w
  for (int i = 0; i < x[0].size(); i++) {
    // Calculate w_i
    for (int j = 0; j < x.size(); j++) {
      w[i] += m[j] * y[j] * x[j][i];
    }
  }
}

vector<double> DualPerceptron::getWeights() {
  return this->w;
}

vector<double> DualPerceptron::getNormalizedWeights() {
  vector<double> normWeights = this->w;
  for (int i = 0; i < normWeights.size(); i++) {
    normWeights[i] /= -this->b;
  }
  return normWeights;
}

double DualPerceptron::getBias() {
  return this->b;
}
