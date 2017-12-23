import math
import unittest
import pandas as pd
import numpy as np
from unittest import mock
import textmining as txm
from testfixtures import tempdir, compare
import os
from ch3 import get_msg, get_msgdir, tdm_df,make_term_df, classify_email
from data import data_dir
from pandas.util.testing import assert_frame_equal, assert_series_equal, assert_index_equal

class TestCh3(unittest.TestCase):


    def setUp(self):
        self.doclist=["I have't eat yet", "The email does not exist, please try again"]

    @mock.patch("builtins.open", create=True)
    def test_get_msg(self,mock_open):
        mock_open.side_effect = [
            mock.mock_open(read_data="\nData1\n").return_value
        ]

        self.assertEqual("Data1\n", get_msg("fileA"))
        mock_open.assert_called_once_with("fileA",'rU', encoding='latin1')

    # @mock.patch("ch3.get_msg")
    @tempdir()
    def test_get_msgdir(self, d):
        get_mock = mock.Mock(return_value=1)
        get_msg = get_mock
        d.write("test.txt", b'\nabc')
        d.write("test2.txt", b'\naaaaaa')
        msg = get_msgdir(d.path)
        # self.assertTrue(mock.called)
        # self.assertEqual(get_mock.call_count,2)
        self.assertEqual(msg, ["abc", 'aaaaaa'])

    def test_tdm_df(self):
        tdm = txm.TermDocumentMatrix()
        for doc in self.doclist:
            tdm.add_doc(doc)
        l = [r for r in tdm.rows(cutoff = 1)]
        df = pd.DataFrame(np.array(l[1:]), columns=l[0])
        result = tdm_df(self.doclist, remove_punctuation=False)
        assert_frame_equal(result, df)

class TestMakeTermDf(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([[1,3],[2,5]], columns=['a','b'])

    def test_make_term_df_index(self):
        result = make_term_df(self.df)
        assert_index_equal(result.columns, pd.Index(['frequency','density','occurrence']))

    def test_make_term_df_frequency(self):
        result = make_term_df(self.df)
        assert_series_equal(result['frequency'], pd.Series([3,8], index=['a','b'], name = 'frequency'))

    def test_make_term_df_density(self):
        result = make_term_df(self.df)
        assert_series_equal(result['density'], pd.Series([3/11,8/11], index=['a','b'], name='density'))

    def test_make_term_df_occurrence(self):
        result = make_term_df(self.df)
        assert_series_equal(result['occurrence'], pd.Series([1.0, 1.0], index=['a','b'], name='occurrence'))


class TestClassifyEmail(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([[1,3],[2,5]], columns=['a','b'])
        self.term_df = make_term_df(self.df)
        self.msg = "cde efg hi"
        self.p = math.log(0.5) + math.log(1e-6)*len(self.msg)

    def test_classify_email_calls_args(self):
        result = classify_email(self.msg,self.term_df)
        self.assertEqual(result,self.p)

if __name__ == "__main__":
    unittest.main()
