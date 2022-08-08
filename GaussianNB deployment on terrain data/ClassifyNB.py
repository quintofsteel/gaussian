def classify(features_train, labels_train):   
    """
    Function takes in the features and labels for the training data and returns the trained classifier
    """
    
    from sklearn.naive_bayes import GaussianNB              ### import the sklearn module for GaussianNB
    clf = GaussianNB()                                      ### create classifier
    clf.fit(features_train, labels_train)                   ### fit the classifier on the training features and labels
    return clf                                              ### return the fit classifier
    
    
