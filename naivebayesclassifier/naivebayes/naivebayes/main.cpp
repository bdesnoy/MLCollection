//
//  main.cpp
//  asn1ex6
//
//  Created by Brian Desnoyers on 2/6/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include <iostream>
#include <vector>
#include "NaiveBayes.hpp"
#include "strtk.hpp"

using namespace std;

// const string TRAINING_SET_FILENAME = "adult.data";
// const string TEST_SET_FILENAME = "adult.test";

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
  
  if (data.size() <= 0) {
    cout << "File at " << filename << " not found." <<
      " Please ensure the working directory is set properly." << endl;
  }
  return data;
}

int main(int argc, const char * argv[]) {
  // Read in the data
  vector<vector<string>> trainingData = readTextFile(argv[1], 0);
  vector<vector<string>> testData = readTextFile(argv[2], 1);
  
  // Create vectors for the featues and classes and create the model
  vector<pair<string, NaiveBayesFeatureType>> features =  { make_pair("age", int_continuous),
                                                            make_pair("workclass", discrete),
                                                            make_pair("fnlwgt", int_continuous),
                                                            make_pair("education", discrete),
                                                            make_pair("education-num", int_continuous),
                                                            make_pair("marital-status", discrete),
                                                            make_pair("occupation", discrete),
                                                            make_pair("relationship", discrete),
                                                            make_pair("race", discrete),
                                                            make_pair("sex", discrete),
                                                            make_pair("capital-gain", int_continuous),
                                                            make_pair("capital-loss", int_continuous),
                                                            make_pair("hours-per-week", int_continuous),
                                                            make_pair("native-country", discrete)
                                                          };
  vector<string> classes = { ">50K", "<=50K" };
  NaiveBayesClassifier incomePredictor = NaiveBayesClassifier(features, classes);
  
  // Train the model
  for (int i = 0; i < trainingData.size(); i++) { //
    vector<string> features = trainingData[i];
    if (features.size() > 0) {
      features.pop_back();
      incomePredictor.addTrainingSample(features, trainingData[i].back());
    }
  }
  incomePredictor.endTraining();
  
  // Test the model on the test data
  for (int i = 0; i < testData.size(); i++) {
    vector<string> features = testData[i];
    if (features.size() > 0) {
      features.pop_back();
      string className = testData[i].back();
      className = className.substr(0, className.size() - 1); // remove period at end
      incomePredictor.testSample(features, className);
    }
  }
  
  // Print the test accuracy
  incomePredictor.printTestAccuracy();
  
  return 0;
}
