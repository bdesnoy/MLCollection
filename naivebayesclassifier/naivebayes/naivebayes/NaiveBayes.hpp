//
//  NaiveBayes.hpp
//  asn1ex6
//
//  Created by Brian Desnoyers on 2/6/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#ifndef IncomePredictor_hpp
#define IncomePredictor_hpp

#include <map>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

double mean(vector<int> values);
double varience(vector<int> values, double mean);

enum NaiveBayesFeatureType { discrete, int_continuous };

class NaiveBayesClassifier {
private:
  // A listing of the features and their types,
  // set by the user before training
  vector<pair<string,NaiveBayesFeatureType>> features;
  
  // The classes to predict, set by the user before training
  vector<string> classes;
  
  // The stored data for each feature, then class (in the order stored in `features`.)
  // For discrete features, this means a frequency count for each class.
  // For continuous features, this means retaining the values for each class.
  
  // feature name -> class name -> category name -> int (count)
  map<string, map<string, map<string, int>>> discreteFeatureData;
  
  // feature name -> class name -> [int (value)]
  map<string, map<string, vector<int>>> continuousFeatureData;
  
  // The count of training samples for each class, and total
  map<string, double> classCount;
  int totalTrainingSamples;
  
  // Values for tracking accuracy
  int testCount;
  int correctCount;
  
  // Summary statistics for continuous features
  map<string, map<string, double>> continuousFeatureMeans;
  map<string, map<string, double>> continuousFeatureVariances;
  
  /**
   Handles a discrete feature when adding a training sample.

   @param featureName the feature name
   @param className the class name
   @param value the discrete value
   */
  void addDiscreteSample(string featureName, string className, string value);
  
  /**
   Handles a continuous (integer) sample when adding a training sample

   @param featureName the feature name
   @param className the class name
   @param value the int value
   */
  void addIntContinousSample(string featureName, string className, int value);
  
  
  /**
   Returns the discrete count of features matching `featureName` in the class with
   name `className`
   
   @param featureName the name of the feature
   @param className the name of the class
   @return the count of the specified feature in the specified class
   */
  int discreteFeatureCount(string featureName, string className);

public:
  
  /**
   Constructor for a new naive bayes classifier with the features described in
   the feature vector and classes described in the class vector.
   
   @param features the features (in order) to be used in training and testing samples
   @param classes  the classes
   */
  NaiveBayesClassifier(vector<pair<string,NaiveBayesFeatureType>> features, vector<string> classes);
  
  /**
   Adds a training sample to the model.
   
   Unknown features should be marked with question marks.

   @param features the features array, in the order specified when creating the class
   @param classLabel the class label of the sample
   */
  void addTrainingSample(vector<string> features, string classLabel);
  
  /**
   Tests the sample with features and classifies it. Compares the classified
   label to the true label `classLabel`.

   @param features the sample features
   @param classLabel the true label
   */
  void testSample(vector<string> features, string classLabel);
  
  /**
   After testing all the samples, this prints the accuracy of the test.
   */
  void printTestAccuracy();
  
  /**
   Ends training- calculates summary statistics for continuous features.
   */
  void endTraining();
  
};

#endif /* IncomePredictor_hpp */
