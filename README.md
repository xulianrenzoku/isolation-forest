# Anomaly Detection Using Isolation Forest

## Project Objective
The objective of this project was to implement the Isolation Forest algorithm as defined in [this paper](https://github.com/xulianrenzoku/isolation-forest/blob/master/IsolationForestPaper.pdf) from scratch.

## Idea
**Method**: Recursively split a certain dataset on a random feature and then split points. (Build an ensemble of binary trees)  

**Motivation**: Anomalies are few and different. Therefore, it is easy to single those out at earlier stage. (Shorter path lengths to the top)  

**Characteristic**: This algorithm does not construct a profile of normal instances and then identify those that do not conform.  

**Advantage**: This algorithm possesses a linear time complexity with a low constant, thus requires less memory.

## Evaluation 
Evaluation of the algorithm was performed using a script provided by Professor Terence Parr of University of San Francisco, which tests for true positive rate (TPR), false positive rate (FPR), and fit time across three labeled datasets.
