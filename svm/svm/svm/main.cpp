//
//  main.cpp
//  svm
//
//  Created by Brian Desnoyers on 2/18/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include <iostream>
#include <vector>
#include "strtk.hpp" // Import STRTK
#include "SimpSVM.hpp"

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
  
  double C = 100.0; // regularization parameter
  double TOL = 0.001; // numerical tolerance
  int MAX_PASSES = 100; // max # of times to iterate over alphas w/o changing
  
  // Read training and test sets
  vector<vector<string>> trainingSet = readTextFile(trainingSetFilename, 1);
  vector<vector<string>> testSet = readTextFile(testSetFilename, 1);
  
  // Create feature and label vectors from ~20% of training set
  vector<vector<int>> features;
  vector<int> labels;
  for (int i = 0; i < trainingSet.size(); i++) {
    if (!(rand() % 5)) {
      vector<int> feature;
      feature.resize(trainingSet[i].size() - 1);
      transform(trainingSet[i].begin() + 1, trainingSet[i].end(), feature.begin(), stringToBinary);
      features.push_back(feature);
      labels.push_back((trainingSet[i][0] == "3") ? 1 : -1);
    }
  }
  
  BinSVM svmClassifier = BinSVM(C, TOL, MAX_PASSES);
  cout << "Beginning training w/ " << features.size() << " random samples..." << endl;
  svmClassifier.train(features, labels);
  cout << "Training complete!" << endl;
  
  // Create feature and label vectors from the test set
  vector<vector<int>> testFeatures;
  vector<int> testLabels;
  for (int i = 0; i < testSet.size(); i++) {
    vector<int> feature;
    feature.resize(testSet[i].size() - 1);
    transform(testSet[i].begin() + 1, testSet[i].end(), feature.begin(), stringToBinary);
    testFeatures.push_back(feature);
    testLabels.push_back((testSet[i][0] == "3") ? 1 : -1);
  }
  
  // Calculate accuracy on test set
  int correctCount = 0;
  for (int i = 0; i < testSet.size(); i++) {
    int yHat = svmClassifier.predictClass(testFeatures[i]);
    if (yHat == testLabels[i]) {
      correctCount++;
    }
  }
  cout << "Accuracy = " << correctCount << "/" << testSet.size() << " = " << ((double)correctCount) / ((double)testSet.size()) << endl;
  
  return 0;
}
