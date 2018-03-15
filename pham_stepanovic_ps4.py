# Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve, auc

np.random.seed(seed=0)
%matplotlib inline


# Load Data

email = pd.read_csv("email.csv")

email.head()
email.tail()
print(email.columns.values)

email.fillna("n", inplace=True)

for em in email["Subject"].tolist():
    if not isinstance(em, str):
        print(em)

len(email)


# Bags of Words
'''As we are working with text data, and models are unable to work with strings as input data, we need to convert them to numbers. Bags of Words will allow us to count and vectorize all words in a given set, then transform the strings into series of numbers that represent the words for the model to work with.'''

corpus = {}

for col in email.columns.values:
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(email[col]).todense()
    corpus[col] = features

print(corpus.keys())

# Term Frequency times Inverse Document Frequenct (tf-idf)

corpus_tfidf = {}
for col in email.columns.values:
    tfidf_transformer = TfidfTransformer()
    corpus_tfidf[col] = tfidf_transformer.fit_transform(corpus[col]).todense()
    print(corpus_tfidf[col].shape)

# Splitting features
'''Here the desired features are split into separate dataframes. This is because each feature, when converted to bags of words, is a matrix of the features and their bags of words. Creating a dataframe where each element is a matrix is impossible, so to get around this I am splitting each desired feature into its own dataframe, and as necessary concatenating dataframes to combine features.'''

subject = pd.DataFrame(corpus_tfidf["Subject"])

body = pd.DataFrame(corpus_tfidf["Body"])

fromname = pd.DataFrame(corpus_tfidf["From: (Name)"])

fromaddress = pd.DataFrame(corpus_tfidf["From: (Address)"])

fromtype = pd.DataFrame(corpus_tfidf["From: (Type)"])

spam = pd.DataFrame(corpus_tfidf["type"])
spam = spam[1]


# Multinomial Naive Bayes model
def multinb(x, y):
    """
    
    This function performs the required functions for fitting and prediction a Multinomial Naive Bayes
    from given x and y datasets.
    
    Args:
        x (array-like): independent data
        y (array-like): target data
        
    Return:
        score (float): Mean accuracy of the model on the given test and target data
    
    """
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
    
    # Fit and predict model
    multinb = MultinomialNB()
    multinb.fit(X_train, y_train)
    
    predicted = multinb.predict(X_test)
    predicted
    
    multinb.predict(X_test)
    score = multinb.score(X_test, y_test)
    
    return score


# Random Forest model
def random_forest(x, y):
    """
    
    This function performs the required functions for fitting and prediction a Logistic Regression model
    from given x and y datasets.
    
    Args:
        x (array-like): independent data
        y (array-like): target data
        
    Return:
        score (float): Mean accuracy of the model on the given test and target data
    
    """
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
    
    # Fit and predict model
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    
    predicted = rf.predict(X_test)
    predicted
    
    rf.predict(X_test)
    score = rf.score(X_test, y_test)
    
    return score

# Logistic regression model
def logisticregression(x, y):
    """
    
    This function performs the required functions for fitting and prediction a Logistic Regression model
    from given x and y datasets.
    
    Args:
        x (array-like): independent data
        y (array-like): target data
        
    Return:
        score (float): Mean accuracy of the model on the given test and target data
    
    """
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
    
    # Fit and predict model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    predicted = logreg.predict(X_test)
    predicted
    
    logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    
    return score

# Results
print("-----Multinomial NB-----")
subjectscore = multinb(subject, spam)
print("Model Mean Accuracy [Subject]:", subjectscore)

bodyscore = multinb(body, spam)
print("\nModel Mean Accuracy [Body]:", bodyscore)

fnamescore = multinb(fromname, spam)
print("\nModel Mean Accuracy [From Name]:", fnamescore)

faddscore = multinb(fromaddress, spam)
print("\nModel Mean Accuracy [From Address]:", faddscore)

ftypescore = multinb(fromtype, spam)
print("\nModel Mean Accuracy [From Type]:", ftypescore)

subbodscore = multinb(pd.concat([subject, body], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body]:", subbodscore)

subbodfnamescore = multinb(pd.concat([subject, body, fromname], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Name]:", subbodfnamescore)

subbodfaddscore = multinb(pd.concat([subject, body, fromaddress], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Address]:", subbodfaddscore)

print("-----Random Forest-----")
subjectscore = random_forest(subject, spam)
print("Model Mean Accuracy [Subject]:", subjectscore)

bodyscore = random_forest(body, spam)
print("\nModel Mean Accuracy [Body]:", bodyscore)

fnamescore = random_forest(fromname, spam)
print("\nModel Mean Accuracy [From Name]:", fnamescore)

faddscore = random_forest(fromaddress, spam)
print("\nModel Mean Accuracy [From Address]:", faddscore)

ftypescore = random_forest(fromtype, spam)
print("\nModel Mean Accuracy [From Type]:", ftypescore)

subbodscore = random_forest(pd.concat([subject, body], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body]:", subbodscore)

subbodfnamescore = random_forest(pd.concat([subject, body, fromname], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Name]:", subbodfnamescore)

subbodfaddscore = random_forest(pd.concat([subject, body, fromaddress], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Address]:", subbodfaddscore)


print("-----Logistic Regression-----")
subjectscore = logisticregression(subject, spam)
print("Model Mean Accuracy [Subject]:", subjectscore)

bodyscore = logisticregression(body, spam)
print("\nModel Mean Accuracy [Body]:", bodyscore)

fnamescore = logisticregression(fromname, spam)
print("\nModel Mean Accuracy [From Name]:", fnamescore)

faddscore = logisticregression(fromaddress, spam)
print("\nModel Mean Accuracy [From Address]:", faddscore)

ftypescore = logisticregression(fromtype, spam)
print("\nModel Mean Accuracy [From Type]:", ftypescore)

subbodscore = logisticregression(pd.concat([subject, body], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body]:", subbodscore)

subbodfnamescore = logisticregression(pd.concat([subject, body, fromname], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Name]:", subbodfnamescore)

subbodfaddscore = logisticregression(pd.concat([subject, body, fromaddress], axis=1), spam)
print("\nModel Mean Accuracy [Subject + Body + From Address]:", subbodfaddscore)


# Roc Curves for each model
# Need the y_test data and prediction probabilities
X_train, X_test, y_train, y_test = train_test_split(pd.concat([subject, body], axis=1), spam,
                                                               test_size = 0.33, random_state = 0)

# Naive Bayes
# Fit and get prediction probabilities
multinb = MultinomialNB()
predict_prob = multinb.fit(X_train, y_train).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, predict_prob[:,1])
roc_auc = auc(fpr, tpr)

fig,ax = plt.subplots(figsize=(10,7))
lw = 2
ax.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Receiver operating characteristic', fontsize=16)
ax.legend(loc="lower right", fontsize=12)
ax.tick_params(axis="both", labelsize=12)


# Random Forest
# Fit and get prediction probabilities
rf = RandomForestClassifier(max_depth=2, random_state=0)
predict_prob = rf.fit(X_train, y_train).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, predict_prob[:,1])
roc_auc = auc(fpr, tpr)

fig,ax = plt.subplots(figsize=(10,7))
lw = 2
ax.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Receiver operating characteristic', fontsize=16)
ax.legend(loc="lower right", fontsize=12)
ax.tick_params(axis="both", labelsize=12)

# Logistic Regression
# Fit and get prediction probabilities
logreg = LogisticRegression()
predict_prob = logreg.fit(X_train, y_train).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, predict_prob[:,1])
roc_auc = auc(fpr, tpr)
fig,ax = plt.subplots(figsize=(10,7))
lw = 2
ax.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Receiver operating characteristic', fontsize=16)
ax.legend(loc="lower right", fontsize=12)
ax.tick_params(axis="both", labelsize=12)