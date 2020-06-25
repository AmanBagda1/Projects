import nltk
import random
import os
from nltk.corpus.reader import CategorizedPlaintextCorpusReader

corpusdir = 'txt_sentoken/' # Directory of corpus.

mycorpus = nltk.corpus.reader.CategorizedPlaintextCorpusReader(r"C:\Users\amanb_2jhwatd\Desktop\Machine Learning\txt_sentoken",r'(?!\.).*\.txt', 
    cat_pattern=r'(neg|pos)/.*',
    encoding="ascii")
mycorpus.words(categories="pos")
print ("Categorize : ", mycorpus.categories()) 
  
print ("\nNegative field : ",mycorpus.fileids(categories =['neg'])) 
  
print ("\nPosiitve field : ", mycorpus.fileids(categories =['pos']))

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents=[(list(mycorpus.words(fileid)),category)
           for category in mycorpus.categories()
           for fileid in mycorpus.fileids(category)]
random.shuffle(documents)
all_words = []
for w in mycorpus.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["stupid"])

#Getting the features from the data
word_features = list(all_words.keys())[:3000]
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
#print((find_features(mycorpus.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

#Naive Bayes Classifier
training_set=featuresets[:1900]
testing_set=featuresets[1900:]
Classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Accuracy Percent: ",nltk.classify.accuracy(Classifier,testing_set))
Classifier.show_most_informative_features(15)

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(Classifier, testing_set))*100)
Classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(Classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
