//
//  main.cpp
//  Perceptron
//
//  Created by Brian Desnoyers on 2/23/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include "Perceptron.hpp"
#include "strtk.hpp" // Import STRTK
#include <iostream>
#include <vector>

using namespace std;

// const string TRAINING_SET_FILENAME = "percep1.txt";
// const string TRAINING_SET_2_FILENAME = "percep2.txt";

double stringToDouble(string str) {
  return stod(str.c_str());
}

/**
 Reads a TSV file and stores it in a 2D vector.
 
 @param filename the filename of the file to read
 @return the 2D vector containing the TSV data
 */
vector<vector<string>> readTextFile(string filename, int skipLines) {
  vector<vector<string>> data;
  string tempLine;
  vector<string> tempRow;
  ifstream infile(filename);
  
  for (int i = 0; i < skipLines; i++) {
    getline(infile, tempLine);
  }
  while (getline(infile, tempLine)) {
    strtk::parse(tempLine, "\t", tempRow);
    data.push_back(tempRow);
    tempRow.clear();
  }
  
  return data;
}

void printWeights(vector<double> weightVector) {
  for (int i = 0; i < weightVector.size(); i++) {
    cout << weightVector[i] << " ";
  }
  cout << endl;
}

int main(int argc, const char * argv[]) {
  
  // Access args
  string trainingSetFilename = argv[1];
  string trainingSet2Filename = argv[2];
  
  // Read training sets
  vector<vector<string>> trainingSet = readTextFile(trainingSetFilename, 0);
  vector<vector<string>> trainingSet2 = readTextFile(trainingSet2Filename, 0);
  
  // Create feature and label vectors from training set #1
  vector<vector<double>> features;
  vector<int> labels;
  for (int i = 0; i < trainingSet.size(); i++) {
    vector<double> feature;
    feature.resize(trainingSet[i].size() - 1);
    transform(trainingSet[i].begin(), trainingSet[i].end() - 1, feature.begin(), stringToDouble);
    features.push_back(feature);
    labels.push_back(atoi(trainingSet[i].back().c_str()));
  }
  
  // Create feature and label vectors from training set #2
  vector<vector<double>> features2;
  vector<int> labels2;
  for (int i = 0; i < trainingSet2.size(); i++) {
    vector<double> feature;
    feature.resize(trainingSet2[i].size() - 1);
    transform(trainingSet2[i].begin(), trainingSet2[i].end() - 1, feature.begin(), stringToDouble);
    features2.push_back(feature);
    labels2.push_back(atoi(trainingSet2[i].back().c_str()));
  }
  
  cout << "Training " << trainingSetFilename << " w/ primal perceptron" << endl;
  Perceptron model;
  model.train(features, labels);
  // Get the weights from the model
  vector<double> modelRawWeights = model.getWeights();
  cout << "Raw weights: ";
  printWeights(modelRawWeights);
  // Get the normalized weights from the model
  vector<double> modelNormWeights = model.getNormalizedWeights();
  cout << "Normalized weights: ";
  printWeights(modelNormWeights);
  cout << endl;
  
  cout << "Training " << trainingSetFilename << " w/ dual-form perceptron (linear kernel)" << endl;
  DualPerceptron dpModel = DualPerceptron(dotProduct);
  dpModel.train(features, labels);
  // Get the weights from the model
  vector<double> dpModelRawWeights = dpModel.getWeights();
  cout << "Raw weights: ";
  printWeights(dpModelRawWeights);
  // Get the normalized weights from the model
  vector<double> dpModelNormWeights = dpModel.getNormalizedWeights();
  cout << "Normalized weights: ";
  printWeights(dpModelNormWeights);
  cout << endl;
  
  cout << "Training " << trainingSet2Filename << " w/ dual-form perceptron (gaussian kernel)" << endl;
  DualPerceptron dpModel2 = DualPerceptron(gaussianKernel);
  dpModel2.train(features2, labels2);
  // Get the weights from the model
  vector<double> dpModelRawWeights2 = dpModel2.getWeights();
  cout << "Raw weights: ";
  printWeights(dpModelRawWeights2);
  // Get the normalized weights from the model
  vector<double> dpModelNormWeights2 = dpModel2.getNormalizedWeights();
  cout << "Normalized weights: ";
  printWeights(dpModelNormWeights2);
  cout << endl;
  
  return 0;
}
