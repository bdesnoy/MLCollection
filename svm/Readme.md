SVM with Simplified SMO Algorithm
==============

This project contains a simple SVM w/ simplified SMO algorithm class along with an example of usage. This example performs binary classification based on image data containing either a handwritten '3' or '5' from [a pre-preocessed dataset from a Kaggle competition](http://www.kaggle.com/c/digit-recognizer/data). 

To run the classifier for training and testing sets:
--------------------------

1.  Compile:
      ```clang++ -std=gnu++11 -stdlib=libc++ strtk.hpp SimpSVM.hpp SimpSVM.cpp main.cpp```

2.  Execute
      ```./a.out [path_to_training_set] [path_to_test_set]```

    ex: with training and test data in same folder:
      ```./a.out train.csv test.csv```