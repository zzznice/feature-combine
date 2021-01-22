# feature-combine
Predicting protein subcellular localization based on machine learning

Data and code are shown above

Purpose: Use machine learning methods to predict the location of protein sub-cells

Data: G_n Gram-negative bacterial protein, G_p Gram-positive bacterial protein in data

step:
1. Perform feature extraction of protein sequences, that is, use the three feature extraction algorithms in feature extraction in the code file, autocorrelation coefficient (ACF_6.m), pseudo amino acid composition (PseAAC.m), g-spaced dipeptide group (Ggap DC.m), this code also improves the autocorrelation coefficient (ACF_7.m). The improvement is to perform factor analysis on the eigenvalues used in ACF_6 and merge them into 7 factors. The improved technology extracts The feature vector of is better after classification.

2.Random fusion of the three feature extraction algorithms. After verification, the three algorithms have the best effect after fusion.

3.classification algorithm. First, perform PCA dimensionality reduction on the multi-dimensional vector after feature fusion, and then use a variety of common classification algorithms KNN, SVM, LR, RF and XGBOOST to compare, and it is found that XGBOOST performs best.

The improved algorithm has a better classification effect in the five classifiers, and the performance of the two data sets after the feature fusion reaches 96.9% and 97.6%, which has obvious reference significance for the localization of protein subcellular.
