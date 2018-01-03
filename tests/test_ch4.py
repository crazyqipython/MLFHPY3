#coding:utf-8
import os
import unittest
from unittest import mock
import shutil
import tempfile
from pandas import *
from pandas.util.testing import assert_frame_equal
from ch4 import easyham_path, parse_email_list, parse_email,\
    get_header_and_message, get_date, get_sender, get_subject,\
    make_email_df,train_test_split,thread_flags, clean_subject
import dateutil.parser as dtp

class TestParseEmailList(unittest.TestCase):

    def test_eashham_pass(self):
        self.assertEqual(easyham_path, "../data/easyham")

    @mock.patch("ch4.parse_email", return_value=3)
    def test_parse_email_list(self, mock):
        result = parse_email_list(['a', 'b','c'])
        self.assertEqual(list(result), [3,3,3])

class TestParseEmail(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir + 'test.txt')
        self.f = open(self.path, 'w')
        self.f.write('aaaa\n\nThe owls are not what they seem')
        self.f.close()

    def tearDown(self):
        shutil.rmtree(self.dir)

    @mock.patch("ch4.get_subject", return_value='d')
    @mock.patch("ch4.get_sender", return_value='c')
    @mock.patch("ch4.get_date", return_value='b')
    @mock.patch("ch4.get_header_and_message", return_value=['a','g'])
    def test_parse_email(self, mock1, mock2, mock3, mock4):
        result = parse_email(self.path)
        self.assertEqual(result[:4],('b','c','d','g'))

    def test_get_header_and_message(self):
        result = get_header_and_message(self.path)
        self.assertEqual((['aaaa\n'], ['The owls are not what they seem']), result)

    def test_get_date(self):
        result = get_date(["Date: 2017-01-12"])
        self.assertEqual(result, dtp.parse('2017-01-12'))

    def test_get_sender(self):
        result = get_sender(["X-Egroups-From: Steve Burt <steve.burt@cursor-system.com> ", "From: Steve Burt <Steve_Burt@cursor-system.com>", "X-Yahoo-Profile: pyruse"])
        self.assertEqual(result, "steve.burt@cursor-system.com")

    def test_get_subject(self):
        result = get_subject(["Subject: [IIU] Eircom aDSL Nat'ing"])
        self.assertEqual(result, "[iiu] Eircom aDSL Nat'ing".lower())


class TestMakeEmailDf(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.df = DataFrame({},
                            columns=['date', 'sender', 'subject',
                                    'message', 'filename'], index=[])
    def test_make_email_df(self):
        result = make_email_df(self.dir)
        assert_frame_equal(result, self.df)


class TestTrainSplit(unittest.TestCase):

    def setUp(self):
        self.df = DataFrame({}, columns=['date', 'sender', 'subject',
                                    'message', 'filename'])

    def test_train_set_split(self):
        result = train_test_split(self.df)
        assert_frame_equal(result, self.df)


class TestThreadFlags(unittest.TestCase):

    def test_thread_flags(self):
        s = "re:asdfs"
        result = thread_flags(s)
        self.assertTrue(result)

    def test_clean_subject(self):
        s = "fw:aaa"
        result = clean_subject(s)
        self.assertEqual(result, 'aaa')