# coding:utf-8

import os
import math
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import textmining as txtm
from pandas import *

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
all_spam = get_msgdir(spam_path)
all_easyham = get_msgdir(easyham_path)
all_easyham = all_easyham[:500]
all_hardham = get_msgdir(hardham_path)

sw = stopwords.words('english')

#stopwords from R tm package
rsw = read_csv("r_stopwords.csv")['x'].values.tolist()

def tdm_df(doclist, stopwords = [], remove_punctuation = True,
           remove_digits = True, sparse_df = False):
    """
    Create a term-document matrix from a list of e-mails.
    Uses the TermDocumentMatrix function in the `textmining` module.
    But, pre-processes the documents to remove digits and punctuation,
    and post-processes to remove stopwords, to match the functionality
    of R's `tm` package.
    """
    tdm = txtm.TermDocumentMatrix()

    for doc in doclist:
        if remove_punctuation == True:
            translator_pun = str.maketrans('','',string.punctuation)
            doc = doc.translate(translator_pun)
        if remove_digits == True:
            translator_digt = str.maketrans('','',string.digits)
            doc = doc.translate(translator_digt)
        tdm.add_doc(doc)

    # Push the TDM data to a list of lists,
    # then make that an ndarray, which then
    # becomes a DataFrame.
    tdm_rows = []
    for row in tdm.rows(cutoff = 1):
        tdm_rows.append(row)

    tdm_array = np.array(tdm_rows[1:])
    tdm_terms = tdm_rows[0]
    df = DataFrame(tdm_array, columns = tdm_terms)

    if len(stopwords) > 0:
        for col in df:
            if col in stopwords:
                del df[col]

    if sparse_df == True:
        df.to_sparse(fill_value = 0)

    return df

spam_tdm = tdm_df(all_spam, stopwords=rsw, sparse_df = True)

def make_term_df(tdm):
    term_df = DataFrame(tdm.sum(), columns = ['frequency'])
    term_df['density'] = term_df.frequency / float(term_df.frequency.sum())
    term_df['occurrence'] = tdm.apply(lambda x: np.sum((x > 0))) / float(tdm.shape[0])

    return term_df.sort_values(by = 'occurrence', ascending = False)

spam_term_df = make_term_df(spam_tdm)

easyham_tdm = tdm_df(all_easyham, stopwords = rsw, sparse_df = True)

easyham_term_df = make_term_df(easyham_tdm)

def classify_email(msg, training_df, prior = 0.5, c = 1e-6):

    msg_tdm = tdm_df([msg])
    msg_freq = msg_tdm.sum()
    msg_match = list(set(msg_freq.index).intersection(set(training_df.index)))
    if len(msg_match) < 1:
        return math.log(prior) + math.log(c) * len(msg_freq)
    else:
        match_probs = training_df.occurrence[msg_match]
        return (math.log(prior) + np.log(match_probs).sum()
                + math.log(c) * (len(msg_freq) - len(msg_match)))

hardham_spamtest = [classify_email(m, spam_term_df) for m in all_hardham]
hardham_hamtest = [classify_email(m, easyham_term_df) for m in all_hardham]
s_spam = np.array(hardham_spamtest) > np.array(hardham_hamtest)

def spam_classifier(msg_list):
    spamprob = [classify_email(m, spam_term_df) for m in msg_list]
    hamprob = [classify_email(m, easyham_term_df) for m in msg_list]
    classifier = np.where(np.array(spamprob) > np.array(hamprob), 'Spam','Ham')
    out_df = DataFrame({
        'pr_spam':   spamprob,
        'pr_ham':  hamprob,
        'classify':classifier
    },columns = ['pr_spam', 'pr_ham', 'classify'])
    return out_df

def class_stats(df):
    return df.classify.value_counts() / float(len(df.classify))

hardham_classify = spam_classifier(all_hardham)
print('Hard Ham Classification Statistics (first set)')
print(class_stats(hardham_classify))

print('Hard Ham (first set) classification data')
print(hardham_classify.head())
print('shape:',hardham_classify.shape)


if __name__ == "__main__":
    import sys
    # a = get_msgdir(hardham2_path)
    # for i in a:
    #     i = i.encode('latin1')
    #     sys.stdout.buffer.write(i)
    # r = rsw
    # print(len(r))
