//
//  NaiveBayes.cpp
//  asn1ex6
//
//  Created by Brian Desnoyers on 2/6/17.
//  Copyright © 2017 Brian Desnoyers. All rights reserved.
//

#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include "NaiveBayes.hpp"
#include "strtk.hpp"

using namespace std;

/**
 A simple function to compute the mean of an int vector.

 @param values the values to average
 @return the mean
 */
double mean(vector<int> values) {
  double mean = 0;
  
  unsigned long size = values.size();
  for (int i = 0; i < values.size(); i++) {
    mean += ((double)values[i]) / ((double)size);
  }
  
  return mean;
}

/**
 A simple function to compute the varience of an int vector.

 @param values the values to find the varience of
 @param mean the mean of the vector
 @return the varience
 */
double varience(vector<int> values, double mean) {
  double var = 0;
  
  for (int i = 0; i < values.size(); i++) {
    var += (pow(values[i] - mean, 2) / ((double)(values.size() - 1)));
  }
  
  return var;
}

////////////////////////////////////////////////////////////////////////////////

// NaiveBayesClassifier

NaiveBayesClassifier::NaiveBayesClassifier(vector<pair<string,NaiveBayesFeatureType>> features, vector<string> classes) {
  this->testCount = 0;
  this->correctCount = 0;
  this->features = features;
  this->classes = classes;
  this->totalTrainingSamples = 0;
}

void NaiveBayesClassifier::addTrainingSample(vector<string> ft, string classLabel) {
  // Require that the features in the sample match those the class is
  // initialized with.
  assert(ft.size() == this->features.size());
  
  // Iterate through the features to train
  for (int f = 0; f < this->features.size(); f++) {
    if (ft[f].compare("?") != 0) { // If the feature is defined (not a '?')...
      if (this->features[f].second == discrete) {
        this->addDiscreteSample(this->features[f].first, classLabel, ft[f]);
      } else if (this->features[f].second == int_continuous) {
        this->addIntContinousSample(this->features[f].first, classLabel, atoi(ft[f].c_str()));
      }
    }
  }
  
  // Increment the class count
  this->classCount[classLabel]++;
  this->totalTrainingSamples++;
}

void NaiveBayesClassifier::addDiscreteSample(string featureName, string className, string value) {
  this->discreteFeatureData[featureName][className][value]++;
}

void NaiveBayesClassifier::addIntContinousSample(string featureName, string className, int value) {
  this->continuousFeatureData[featureName][className].push_back(value);
}

void NaiveBayesClassifier::testSample(vector<string> ft, string classLabel) {
  
  // We are maximizing probability across the class labels here
  double maxProbability = 0;
  string classLabelForMaxProbability;
  
  // Iterate through the class labels
  for (int c = 0; c < this->classes.size(); c++) {
    double probability = 1;
    string currentClass = this->classes[c];
    
    // Iterate through features
    for (int f = 0; f < this->features.size(); f++) {
      if (ft[f].compare("?") != 0) { // known feature- question marks are skipped
        if (this->features[f].second == discrete) {
          // Category count in this class for this feature
          int countInClassWithThisCategory = this->discreteFeatureData[this->features[f].first][currentClass][ft[f]];
          // Total count in this class for this feature
          int totalCountInClass = this->discreteFeatureCount(this->features[f].first, currentClass);
          
          probability *= ((double) countInClassWithThisCategory) / ((double) totalCountInClass);
        } else if (this->features[f].second == int_continuous) {
          double avg = this->continuousFeatureMeans[this->features[f].first][currentClass];
          double var = this->continuousFeatureVariances[this->features[f].first][currentClass];
          int value = atoi(ft[f].c_str());
          // Calculate probability w/ Gaussian distribution
          probability *= ((1.0) / sqrt(2 * M_PI * var)) * pow(M_E, -1 * (pow(value - avg, 2) / (2 * var)));
        }
      }
    }
    
    // Multiply probability times the probability of the class label
    probability *= ((double)this->classCount[currentClass])/((double)this->totalTrainingSamples);
    
    cout << currentClass << ": P =" <<  probability << endl;
    
    // Update max probability
    if (probability > maxProbability) {
      maxProbability = probability;
      classLabelForMaxProbability = currentClass;
    }
  }
  
  this->testCount++;
  if (classLabelForMaxProbability == classLabel) {
    cout << "[Correct] Predicted class: " << classLabelForMaxProbability << " with P = " << maxProbability << endl;
    this->correctCount++;
  } else {
    cout << "[Wrong]   Predicted class: " << classLabelForMaxProbability << " with P = " << maxProbability << " Expected: " << classLabel << endl;
  }
}

int NaiveBayesClassifier::discreteFeatureCount(string featureName, string className) {
  map<string, int> counts = this->discreteFeatureData[featureName][className];
  
  int result = 0;
  for(auto const& count : counts) {
    result += count.second;
  }
  
  return result;
}

void NaiveBayesClassifier::endTraining() {
  for(auto const& outerdict : this->continuousFeatureData) {
    string featureString = outerdict.first;
    for(auto const& innerdict : outerdict.second) {
      string classString = innerdict.first;
      vector<int> values = innerdict.second;
      
      double avg = mean(values);
      this->continuousFeatureMeans[featureString][classString] = avg;
      this->continuousFeatureVariances[featureString][classString] = varience(values, avg);
    }
  }
}

void NaiveBayesClassifier::printTestAccuracy() {
  cout << "Model Accuracy on Test Data: " << this->correctCount << "/" << this->testCount << " = " << (double)this->correctCount / (double)this->testCount << endl;
}
