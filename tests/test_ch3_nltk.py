import unittest
from unittest import mock
from ch3_nltk import (get_msg_words, features_from_messages, \
    word_indicator, make_train_test_set,check_classifier)
from nltk.corpus import stopwords


class TestGetMSG(unittest.TestCase):

    def setUp(self):
        self.msg = "3D<head>I've a dog</head>, he's name is jim___scott, he's cute."

    def test_get_msg_for_stopwords_none(self):
        result = get_msg_words(self.msg)
        self.assertListEqual(sorted(result), sorted(['dog', 'he', 've', 'head', 'cute', 'name', 'jim_scott', 'is']))

    def test_get_msg_for_stopwords(self):
        result = get_msg_words(self.msg, stopwords.words('english'))
        self.assertListEqual(sorted(result), sorted(['dog', 'head', 'cute', 'name', 'jim_scott']))

    def test_get_msg_strip_html(self):
        result = get_msg_words(self.msg, stopwords.words('english'), strip_html=True)
        self.assertListEqual(sorted(result), sorted(['dog', 'cute', 'name', 'jim_scott']))


class TestFeaturesFromMessages(unittest.TestCase):

    def setUp(self):
        self.msg = ["I", 'love', 'programming']

    def test_features_from_message(self):
        mock1 = mock.Mock(return_value="I")
        result = features_from_messages(self.msg, 'spam', mock1)
        self.assertListEqual(result, [('I', 'spam'), ('I', 'spam'), ('I', 'spam')])

    def test_word_indicator(self):
        result = word_indicator("I love programming")
        self.assertEqual(result,{'love':True, 'programming':True})


class TestTrainTestSet(unittest.TestCase):
    @mock.patch("ch3_nltk.features_from_messages", return_value=2)
    def test_train_test(self, mock):
        mock_1 = mock.Mock(return_value=1)
        result = make_train_test_set(mock_1)
        self.assertEqual(result, (4,2,2,2))


class TestCheckClassifier(unittest.TestCase):

    def setUp(self):
        self.feature_extract = lambda x: x

    def test_check_classifier(self):
        result = check_classifier(self.feature_extract)
        self.assertEqual(result,None)