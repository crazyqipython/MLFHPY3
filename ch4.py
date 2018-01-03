"""This program indentifies 'priority' emails in a dataset of
 e-mails. Email i is a priority email if rank(i) > threshod,
 where Rank(i) is the product of 5 weighting factors. so if
 w(i) > threshold then email i is a priority message.
 The five weighting functions are: 1.Sender weight 2.thread sender
 weight. 3. Thread activity weight 4.subject term weight. 5
 message term weight"""

import os
import re
import math
import random
import numpy as np
import datetime as dt
import dateutil.parser as dtp
import matplotlib.pyplot as plt
import textmining as txtm
import string
from pandas import *
import sys
from statsmodels.nonparametric import kde
from data import data_dir

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


easyham_path = os.path.abspath(os.path.join(data_dir, 'easy_ham'))

def parse_email_list(file_list):
    for f in file_list:
        yield parse_email(os.path.join(easyham_path, f))


def parse_email(path):
    """Get important elements of an email"""
    file_name = os.path.split(path)[-1]
    header, message = get_header_and_message(path)
    date = get_date(header)
    sender = get_sender(header)
    subj = get_subject(header)
    return(date, sender, subj, message, file_name)


def get_header_and_message(path):
    with open(path, 'rU', encoding='latin1') as con:
        email = con.readlines()
        first_blank_index = email.index('\n')
        header = email[:first_blank_index]
        message = ''.join(email[(first_blank_index + 1): ])
    return header, message


date_pattern = re.compile('^Date')
def get_date(header):
    dateline = [l for l in header if
                re.search(date_pattern, l) != None][0]
    dateline = dateline[6:].strip()
    return dtp.parse(dateline)

splitfrom_pattern = re.compile('[\"\:<> ]')
def get_sender(header):
    sender = list(filter(lambda x: x.find('From: ') != -1, header))[0]
    sender = re.split(splitfrom_pattern, sender)
    sender = filter(lambda s: s != ' ' and s != '', sender)
    sender = list(filter(lambda x: '@' in x, sender))[0]
    return sender.lower().rstrip()


def get_subject(header):
    subject = filter(lambda x: x.find('Subject: ') != -1, header)
    subject = list(subject)
    if len(subject) > 0:
        subject_start = subject[0].find('Subject: ') + 9
        subject = subject[0][subject_start:]
        return subject.lower()
    else:
        return subject

def make_email_df(email_dir):
    """parse each email in a directory and return
    a dataframe of each email
    """
    email_dict = {
        'date':    [],
        'sender':  [],
        'subject': [],
        'message':[],
        'filename':[]
    }
    file_list = os.listdir(email_dir)
    file_list = [f for f in file_list if f != 'cmds']

    parsed_emails = parse_email_list(file_list)
    # Fill up the dictionary with the generator
    for pe in parsed_emails:
        date, sender, subject, message, filename = pe
        email_dict['date'].append(date)
        email_dict['sender'].append(sender)
        email_dict['subject'].append(subject)
        email_dict['message'].append(message)
        email_dict['filename'].append(filename)

    email_df = DataFrame(email_dict,
                         columns = ['date', 'sender', 'subject',
                                    'message', 'filename'])
    return email_df

email_df = make_email_df(easyham_path)
# print(email_df.head())

def train_test_split(df, train_fraction = .5, shuffle = True,
                     preserve_index = True, seed = None):
    if seed:
        random.seed(seed)
    nrows = df.shape[0]
    split_point = int(train_fraction * nrows)
    rows = list(range(nrows))

    if shuffle:
        random.shuffle(rows)

    train_rows = rows[:split_point]
    test_rows  = rows[split_point:]

    train_index = df.index[train_rows]
    test_index = df.index[test_rows]

    train_df = df.ix[train_index, :]
    test_df = df.ix[test_index,:]

    if not preserve_index:
        train_df.index = np.arange(train_df.shape[0])
        test_df.index  = np.arange(test_df.shape[0])
    return train_df, test_df

email_df['sort_date'] = email_df['date'].map(lambda d:
     dt.datetime(d.year,d.month,d.day,d.hour,d.minute,d.second))

email_df = email_df.sort_values('sort_date')
train_df, test_df = train_test_split(email_df, shuffle = False,
                                     preserve_index = False,
                                     seed = 224)

def get_sender_weights(email_df):
    freq = email_df['sender'].value_counts()
    freq.sort_values()
    sender_weights = DataFrame({"freq":freq,
                                'weight': np.log(1.0 + freq)})
    sender_weights = sender_weights.sort_values('weight', ascending=False)
    return sender_weights


sender_weights = get_sender_weights(train_df)
sender_weights_test = get_sender_weights(test_df)
# nshow = 30
# top_emails = sender_weights[:nshow].index

# plt.figure(figsize=(6,14))
# plt.barh(np.arange(nshow), sender_weights['freq'][top_emails],align='center',
#          left=sender_weights['freq'][top_emails].fillna(0),
#          fc    = 'orange',
#          alpha = .8,
#          label = 'Test')
# plt.ylim((0 - .5, nshow - .5))
# plt.title('Frequency of top %i sender addresses' % nshow)
# plt.yticks(np.arange(nshow), top_emails)
# plt.legend(frameon = False)
# plt.savefig('sender_weight.png')

reply_pattern = "(re:|re\[\d\]:)"
fwd_pattern = "(fw:|fw\[\d\]:)"
def thread_flag(s):
    """
    Returns true if string s mathchs the thread patterns
    If s is a pandas series returns a Series of booleans.
    """
    if isinstance(s, str):
        return re.search(reply_pattern, s) is not None
    else:
        return s.str.contains(reply_pattern, re.I)

def clean_subject(s):
    """
    Removes all the reply and forward labeling
    from a string
    If s is a pandas series return a Series of cleaned
    strings.
    This will help find the initial message in the thread
    """
    if isinstance(s,str):
        s_clean = re.sub(reply_pattern, '', s, re.I)
        s_clean = re.sub(fwd_pattern, '', s_clean, re.I)
        s_clean = s_clean.strip()
    else:
        s_clean = s.str.replace(reply_pattern, '', re.I)
        s_clean = s_clean.str.replace(fwd_pattern, '', re.I)
        s_clean = s_clean.str.strip()
    return s_clean


def get_thread_df(email_df):
    """Indentify threads in an email Dataframe,
       and extract them into a new DataFrame.
    """
    # Find threads by emails with reply patterns in their subject.
    # Then get a set of thread subjects.
    is_thread = thread_flag(email_df['subject'])
    thread_subj = email_df['subject'][is_thread]

    # Clean up the subjects by removing reply and forward labels
    thread_subj = clean_subject(thread_subj)
    thread_subj = thread_subj.unique()

    # Search for these thread subjects in the original
    # email DataFrame
    # Prepare the DataFrame for searching by clean up
    # the subjects
    search_df = email_df[['date', 'sender', 'subject']]
    search_df['subject'] = clean_subject(search_df['subject'])

    # Find subject mathchs
    thread_matches = [subj in thread_subj for subj in search_df['subject']]
    mathch_df = search_df.ix[thread_matches, :]

    return mathch_df

thread_df = get_thread_df(train_df)
thread_sender_weights = get_sender_weights(thread_df)
# print(thread_df.head())

def get_thread_activity(thread_df):
    """
    Compute 'activity' statistics on threads in a DataFrame:
    frequency: Number of emails observed in the thread
    span: Time before the first email and last email observed
    weight: Number emails in the thread
    """
    clean_times = thread_df['date'].map(lambda t: t.tzinfo is not None)
    thread_df_clean = thread_df.ix[clean_times, :]

    freq_by_thread = thread_df['subject'].value_counts()

    seconds_span = lambda x: (
        (np.max(x) - np.min(x)).total_seconds())
    span_by_thread = thread_df_clean.groupby('subject')
    span_by_thread = span_by_thread['date'].aggregate(seconds_span)

    activity = DataFrame({
        'freq':freq_by_thread,
        'span': span_by_thread,
        'weight': 10 + np.log10(freq_by_thread/span_by_thread)
    })

    activity = activity[activity['freq'] >= 2]
    activity = activity[notnull(activity['weight'])]
    return activity

thread_activity_df = get_thread_activity(thread_df)

print(thread_activity_df.head())

threads_to_check = ['please help a newbie compile mplayer :-)',
                    'prob. w/ install/uninstall',
                    'http://apt.nixia.no/']
print(thread_activity_df.ix[threads_to_check, :])

rsw = read_csv('r_stopwords.csv')['x'].values.tolist()

def get_thread_subject_term_weights(thread_activity_df):
    """
    Creates a term->weight map based on a DataFrame containing
    threaded subjects and thire activity weights
    """
    thread_subjects = thread_activity_df.index
    thread_tdm = tdm_df(thread_subjects, remove_punctuation = False,
                                         remove_digits = False,
                                         stopwords = rsw)
    def calc_term_weight(term):
        thread_with_term = np.where(thread_tdm[term] > 0.0)
        weight_vec = thread_activity_df['weight'].ix[thread_with_term]
        return weight_vec.mean()
    term_weights = Series({t: calc_term_weight(t) for t in thread_tdm})
    return term_weights

thread_subject_terms_weights = \
    get_thread_subject_term_weights(thread_activity_df)
print(thread_subject_terms_weights.head())

def get_weight_from_terms(term_list, weight_df, subject = False):
    '''
    Given a term-list from an e-mail's message, and a term->weights
    map contained in a DataFrame, calculate the e-mail's message or
    subject term-weight. (default is message)
    '''
    if isinstance(term_list, str):
        term_list = [term_list]

    if subject:
        weights = weight_df
    else:
        weights = weight_df['weight']

    term_list_weight = weights[term_list].mean()

    if np.isnan(term_list_weight):
        term_list_weight = 1.0

    return term_list_weight

def get_message_term_weights(email_df):
    '''
    Creates a term->weight map for terms in the messages of the training
    e-mail DataFrame
    '''
    messages = email_df['message']
    term_freq = tdm_df(messages, stopwords = rsw).sum()

    term_weight_df = DataFrame({'freq'   : term_freq,
                                'weight' : np.log10(term_freq)})

    return term_weight_df

message_terms_weights = get_message_term_weights(train_df)

def rank_email(email_df, row):
    '''
    Ranks an e-mail (as contained in the row of a DataFrame) by
    computing and combining its five weights.
    '''
    email   = email_df.ix[row, :]
    date    = email['date']
    sender  = email['sender']
    subject = email['subject']
    message = email['message']

    # 1. Get sender weights (all messages)
    sender_weight = (sender_weights['weight']
                     .get(sender) or 1.0)

    # 2. Get sender weights (threads)
    thread_sender_weight = (thread_sender_weights['weight']
                           .get(sender) or 1.0)

    # 3. Get thread activity weight
    is_thread = thread_flag(subject)
    subject = clean_subject(subject)
    if is_thread:
        activity_weight = (thread_activity_df['weight']
                           .get(subject) or 1.0)
    else:
        activity_weight = 1.0

    # 4. Get subject line weight via thread-subject term weights
    subj_terms = tdm_df(subject, remove_punctuation = False,
                                 remove_digits = False,
                                 stopwords = rsw).columns
    subj_term_weight = get_weight_from_terms(subj_terms,
                            thread_subject_terms_weights,
                            subject = True)

    # 5. Message term weight
    message_terms = tdm_df(message, stopwords = rsw).columns
    message_term_weight = get_weight_from_terms(message_terms,
                               message_terms_weights)
    weights = [sender_weight,
               thread_sender_weight,
               activity_weight,
               subj_term_weight,
               message_term_weight]

    # The e-mail's final rank is just the product of the weights.
    rank = np.array(weights).prod()

    return rank

def make_rank_df(email_df):
    '''
    Rank each e-mail in a DataFrame.
    '''
    n_emails = email_df.shape[0]

    sender_weight_results = []
    thread_sender_weight_results = []
    activity_weight_results = []
    subj_term_weight_results = []
    message_term_weight_results = []

    rank_results = []
    rank_df = email_df.copy()

    for e in range(n_emails):
        weights_rank = rank_email(email_df, e)
        rank_results.append(weights_rank)

    rank_df['rank'] = rank_results

    return rank_df

train_ranks = make_rank_df(train_df)
print(train_ranks)

threshold = train_ranks['rank'].median()
test_ranks = make_rank_df(test_df)

train_kde = kde.KDEUnivariate((train_ranks['rank']))
train_kde.fit()
test_kde = kde.KDEUnivariate(test_ranks['rank'])
test_kde.fit()

plt.figure(figsize=(8, 6))
plt.fill(train_kde.support, train_kde.density, color = 'steelblue', alpha = .7,
         label = 'Train')
plt.fill(test_kde.support, test_kde.density, color = 'red', alpha = .7,
         label = 'Test')
plt.xlim(0, 400)
plt.ylim(0, np.max(test_kde.density))
plt.axvline(threshold, linestyle = '--', label = 'Priority threshold')
plt.xlabel('Rank')
plt.ylabel('Density')
plt.title('Distribution of ranks for training and test e-mails')
plt.legend(frameon = False)
plt.savefig("rank.png")