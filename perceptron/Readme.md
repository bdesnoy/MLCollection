Perceptron Classifier
==============

This project contains both a primal and dual form of the perceptron along with an example of usage. The dual form allows the use of non-linear kernel functions. This example applies both perceptron implementations to an example linearly seperable dataset ```percep1.txt``` (linear kernel function).  It then applies the dual form of the perceptron w/ a Gaussian kernel to the non-linearly seperable dataset ```percep2.txt```.

Note: This example uses the STRTK library for splitting the CSV data files.

To run the classifier for testing sets:
--------------------------

1.  Compile:
      ```clang++ -std=gnu++11 -stdlib=libc++ strtk.hpp Perceptron.hpp Perceptron.cpp main.cpp```

2.  Execute
      ```./a.out [path_to_training_set1] [path_to_training_set2]```

    ex: with training and test data in same folder:
      	```./a.out percep1.txt percep2.txt```
