//
//  main.cpp
//  multilayerPerceptron
//
//  Created by Brian Desnoyers on 2/22/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include <iostream>
#include "strtk.hpp" // Import STRTK
#include "Perceptron.hpp"

using namespace std;

//const string TRAINING_SET_FILENAME = "train.csv";
//const string TEST_SET_FILENAME = "test.csv";

/**
 Converts a string `str` to an int.

 @param str the string to convert
 @return the int value of the string
 */
double stringToInt(string str) {
  return stoi(str.c_str());
}

/**
 Converts a string `str` to a binary int.

 @param str the string to convert
 @return the number 1 if the int value of the string is > 0
 */
double stringToBinary(string str) {
  return (stringToInt(str) > 0) ? 1 : 0;
}

/**
 Reads a CSV file and stores it in a 2D vector.
 
 @param filename the filename of the file to read
 @return the 2D vector containing the CSV data
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
    strtk::parse(tempLine, ", ", tempRow);
    data.push_back(tempRow);
    tempRow.clear();
  }
  
  return data;
}

int main(int argc, const char * argv[]) {
  
  // Access args
  string trainingSetFilename = argv[1];
  string testSetFilename = argv[2];
  int hiddenNodeCount = atoi(argv[3]);
  
  // Read training and test sets
  vector<vector<string>> trainingSet = readTextFile(trainingSetFilename, 1);
  vector<vector<string>> testSet = readTextFile(testSetFilename, 1);
  
  // Create feature and label vectors from training set
  vector<vector<int>> features;
  vector<int> labels;
  for (int i = 0; i < trainingSet.size(); i++) { // trainingSet.size()
    vector<int> feature;
    feature.resize(trainingSet[i].size() - 1);
    transform(trainingSet[i].begin() + 1, trainingSet[i].end(), feature.begin(), stringToBinary);
    features.push_back(feature);
    labels.push_back((trainingSet[i][0] == "3") ? 0 : 1);
  }
  
  // Set parameters for the MLP
  int instanceCount = (int)features[0].size();  // The number of feature instances
  int classes = 2;                              // The count of classes
  //int hiddenNodeCount = 10;                     // The number of hidden nodes in the hidden layer
  double learningRate = 0.05;                   // The learning rate 0.04
  double momentum = 0.4;                        // The momentum term for gradient descent 0.4
  double leastMeanSquareError = 15;             // The stopping condition
  
  // Create and train the model
  SHLMLP model = SHLMLP(instanceCount, classes, hiddenNodeCount, learningRate, momentum, leastMeanSquareError);
  cout << "Training single hidden layer MLP w/ " << hiddenNodeCount << " hidden nodes." << endl;
  cout << "Stopping when MSE falls < " << leastMeanSquareError << endl;
  model.train(features, labels);
  cout << "Training Complete!" << endl << endl;
  
  // Create feature and label vectors from test set
  vector<vector<int>> testFeatures;
  vector<int> testLabels;
  for (int i = 0; i < testSet.size(); i++) { // testSet.size()
    vector<int> feature;
    feature.resize(testSet[i].size() - 1);
    transform(testSet[i].begin() + 1, testSet[i].end(), feature.begin(), stringToBinary);
    testFeatures.push_back(feature);
    testLabels.push_back((testSet[i][0] == "3") ? 0 : 1);
  }
  
  // Calculate and print the accuracy for the model on the test set
  cout << "Determining Accuracy on Test Set" << endl;
  int correct = 0;
  for (int i = 0; i < testFeatures.size(); i++) {
    int estimate = model.test(testFeatures[i]);
    if (estimate == testLabels[i]) {
      correct++;
    }
  }
  cout << "Accuracy = " << correct << "/" << (int)testFeatures.size() << " = " << (double)correct / (double)testFeatures.size() << endl;
  
  return 0;
}
