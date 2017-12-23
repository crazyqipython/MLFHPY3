from pandas import *
import numpy as np
import os
import re
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import defaultdict


from data import data_dir

spam_path = os.path.join(data_dir, 'spam')
spam2_path = os.path.join(data_dir, 'spam_2')
easyham_path = os.path.join(data_dir, 'easy_ham')
easyham2_path = os.path.join(data_dir, 'easy_ham_2')
hardham_path = os.path.join(data_dir, 'hard_ham')
hardham2_path = os.path.join(data_dir, 'hard_ham_2')
rsw_path = os.path.join(data_dir, 'r_stopwords.csv')

def get_msg(path):
    """
    Read in the `message` portion of an e-mail, given
    its file path. The `message` text begins after the first
    blank line; above is header information.

    Returns a string.
    """
    with open(path, "rU",encoding='latin1') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1):]
        return ''.join(msg)


def get_msgdir(path):
    """
    Read all messages from files in a directory into
    a list where each item is the text of a message.

    Simply gets a list of e-mail files in a directory,
    and iterates `get_msg()` over them.

    Returns a list of strings.
    """
    filelist = os.listdir(path)
    filelist = filter(lambda x: x != 'cmds', filelist)
    # print(len(list(filelist)))
    all_msgs =[get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs

#Get lists containing messages of each type.
train_spam_messages = get_msgdir(spam_path)
train_easyham_messages = get_msgdir(easyham_path)
train_easyham_messages = train_easyham_messages[:500]
train_hardham_messages = get_msgdir(hardham_path)

test_spam_messages    = get_msgdir(spam2_path)
test_easyham_messages = get_msgdir(easyham2_path)
test_hardham_messages = get_msgdir(hardham2_path)

def get_msg_words(msg, stopwords=[], strip_html = False):
    """get msg workds"""
    msg = re.sub('3D', '', msg)

    if strip_html:
        msg = re.sub('<(.|\n)*?>', ' ', msg)
        msg = re.sub('&\w+;', ' ', msg)

    msg = re.sub('_+', '_', msg)

    msg_words = set(wordpunct_tokenize(msg.replace('=\n', '').lower()))

    # Get rid of stopwords
    msg_words = msg_words.difference(stopwords)

    # Get rid of punctuation tokens, numbers, and single letters.
    msg_words = [w for w in msg_words if re.search('[a-zA-Z]', w) and len(w) > 1]

    return msg_words


def features_from_messages(messages, label, feature_extractor, **kwargs):
    features_labels = []
    for msg in messages:
        features = feature_extractor(msg, **kwargs)
        features_labels.append((features, label))
    return features_labels

def word_indicator(msg, **kwargs):
    features = defaultdict(list)
    msg_words = get_msg_words(msg, **kwargs)
    for w in msg_words:
            features[w] = True
    return features

def make_train_test_set(feature_extractor, **kwargs):

    train_spam = features_from_messages(train_spam_messages, 'spam',
                                        feature_extractor, **kwargs)
    train_ham = features_from_messages(train_easyham_messages, 'ham',
                                       feature_extractor, **kwargs)
    train_set = train_spam + train_ham

    test_spam = features_from_messages(test_spam_messages, 'spam',
                                       feature_extractor, **kwargs)
    test_ham = features_from_messages(test_easyham_messages, 'ham',
                                      feature_extractor, **kwargs)

    test_hardham = features_from_messages(test_hardham_messages, 'ham',
                                          feature_extractor, **kwargs)

    return train_set, test_spam, test_ham, test_hardham


def check_classifier(feature_extractor, **kwargs):

    train_set, test_spam, test_ham, test_hardham = \
        make_train_test_set(feature_extractor, **kwargs)

    classifier = NaiveBayesClassifier.train(train_set)
    print('Test Spam accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_spam)))
    print ('Test Ham accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_ham)))
    print ('Test Hard Ham accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_hardham)))
    print(classifier.show_most_informative_features(20))

check_classifier(word_indicator, stopwords = stopwords.words('english'), strip_html=True)