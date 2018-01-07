from pandas import *
import numpy as np
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
import sklearn.cross_validation as cv
from sklearn.svm import l1_min_c
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
import scipy.linalg as la
from math import pi
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrix
import random
import string
import re
import os
import math
import textmining as txtm
from data import data_dir

sin_data = DataFrame({'x': np.linspace(0,1,101)})
sin_data['y'] = np.sin(2 * pi * sin_data['x']) + np.random.normal(0, 0.1, 101)

sin_linpred = ols('y ~x', data=sin_data).fit().predict()
print(sin_linpred)

plt.figure(figsize=(8,8))
plt.plot(sin_data['x'], sin_data['y'], 'k')
plt.plot(sin_data['x'], sin_linpred, 'r-', label='Linear model prediction')
plt.title('Sin wave with gaussion noise')
plt.legend(loc='upper left')
plt.savefig('sin_lin.png')

# Orthonormal polynomial
x = sin_data['x']
y = sin_data['y']
Xpoly = dmatrix('C(x,Poly)')
Xpoly1 = Xpoly[:, :2]
Xpoly3 = Xpoly[:, :4]
Xpoly5 = Xpoly[:, :6]
Xpoly25 = Xpoly[:, :26]

polypred1 = sm.OLS(y, Xpoly1).fit().predict()
polypred3 = sm.OLS(y, Xpoly3).fit().predict()
polypred5 = sm.OLS(y, Xpoly5).fit().predict()
polypred25 = sm.OLS(y, Xpoly25).fit().predict()

plt.figure(figsize=(10,8))
fig, ax = plt.subplots(2,2,sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.0, wspace=0.0)
fig.suptitle('Polynomial fits to noisy sin data', fontsize=16.0)

for a, p, d in zip(ax.ravel(), [polypred1, polypred3, polypred5, polypred25], ['1', '3', '5', '25']):
    a.plot(x, y, '.', color='steelblue', alpha=0.5)
    a.plot(x, p)
    a.text(0.5, 0.95, 'D = '+ d, fontsize=12,
           verticalalignment = 'top',
           horizontalalignment = 'center',
           transform = a.transAxes
           )
    a.grid()

plt.setp(fig.axes[2].get_yaxis().get_ticklabels(), visible = False)
plt.setp(fig.axes[3].get_yaxis(), ticks_position = 'right')
plt.setp(fig.axes[1].get_xaxis(), ticks_position = 'top')
plt.setp(fig.axes[3].get_xaxis().get_ticklabels(), visible = False)
plt.savefig("polynomial_sin.png")

# Poly test
def poly(x, degree):
    '''
    Generate orthonomal polynomial basis function from vector
    :param x:
    :param degree:
    :return:
    '''
    xbar = np.mean(x)
    X = np.power.outer(x - x.mean(), range(0, degree+1))
    Q, R = la.qr(x)
    diagind = np.subtract.outer(range(R.shape[0]), range(R.shape[1])) == 0
    z = R * diagind
    Qz = np.dot(Q, z)
    norm2 = (Qz**2).sum(axis=0)
    Z = Qz/np.sqrt(norm2)
    Z = Z[:, 1:]
    return Z

np.random.seed(1)
rand_ind = list(range(100))
np.random.shuffle(rand_ind)
train_ind = rand_ind[:50]
test_ind = rand_ind[50:]

lasso_model = LassoCV(cv=15, copy_X = True, normalize=True)
lasso_fit = lasso_model.fit(Xpoly[:, 1:11], y)
lasso_path = lasso_model.score(Xpoly[:, 1:11], y)

plt.figure(figsize=(8,8))
plt.plot(-np.log(lasso_fit.alphas_), np.sqrt(lasso_fit.mse_path_).mean(axis=1))
plt.ylabel('RMSE(avg. across folds)')
plt.xlabel(r'$-\log(\lambda)$')
plt.axvline(-np.log(lasso_fit.alpha_), color='red')
plt.savefig("rmse.png")

print(Series(np.r_[lasso_fit.intercept_, lasso_fit.coef_]))

plt.plot(x, y, '.')
plt.plot(np.sort(x), lasso_fit.predict(Xpoly[:, 1:11])[np.argsort(x)], '-r', label='Training data lasso fit')
plt.savefig('training_data.png')

# Predict book sales
book_path = os.path.join(data_dir, 'oreilly.csv')
ranks_df = read_csv(book_path,encoding='latin1')
ranks_df.columns = ['ipfamily', 'title', 'isbn', 'rank', 'long_desc']
ranks = ranks_df['rank']
rank_rev_map = {i:j for i, j in zip(ranks, ranks[::-1])}
ranks.replace(rank_rev_map)

docs = ranks_df['long_desc']
tag_pattern = r'<.*?>'
docs_clean = docs.str.replace(tag_pattern, '')

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

rsw = read_csv(os.path.join(data_dir,'r_stopwords.csv'), squeeze = True).tolist()
desc_tdm = tdm_df(docs_clean, stopwords = rsw)

lasso_model = LassoCV(cv = 10)
lasso_fit = lasso_model.fit(desc_tdm.values, ranks.values)

plt.figure(figsize=(8,8))
plt.plot(-np.log(lasso_fit.alphas_), np.sqrt(lasso_fit.mse_path_), alpha = .5)
plt.plot(-np.log(lasso_fit.alphas_), np.sqrt(lasso_fit.mse_path_).mean(axis = 1),
         lw = 2, color = 'black')
plt.ylim(0, 60)
plt.xlim(0, np.max(-np.log(lasso_fit.alphas_)))
plt.title('Lasso regression RMSE')
plt.xlabel(r'$-\log(\lambda)$')
plt.ylabel('RMSE (and avg. across folds)')
plt.savefig('RMSE_FOLDS.png')

top50 = 1*(ranks_df['rank'] <=50)
min_l1_C = l1_min_c(desc_tdm.values, top50.values)
cs = min_l1_C * np.logspace(0, 3, 10)

cdict = {}
for c in cs:
    cdict[c] = []
print(cdict)

desc_tdm['Top50'] = top50
logit_model = LogisticRegression(C = 1.0, penalty='l1', tol=1e-6)

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
    test_index  = df.index[test_rows]

    train_df = df.ix[train_index, :]
    test_df  = df.ix[test_index, :]

    if not preserve_index:
        train_df.index = np.arange(train_df.shape[0])
        test_df.index  = np.arange(test_df.shape[0])

    return train_df, test_df


np.random.seed(4)
for i in range(50):
    train_docs, test_docs = train_test_split(desc_tdm, 0.8)
    trainy, testy = train_docs.pop('Top50'), test_docs.pop('Top50')
    trainX, testX = train_docs, test_docs

    for c in cdict:
        logit_model.set_params(C = c)
        logit_fit =logit_model.fit(trainX.values, trainy.values)
        predy = logit_fit.predict(testX.values)
        error_rate = np.mean(predy!=testy)
        cdict[c].append(error_rate)

plt.figure(figsize=(8,8))
error_path = DataFrame(cdict).mean()
error_path.plot(style = 'o-k', label = 'Error rate')
error_path.cummin().plot(style='r-', label='Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.savefig('l1_log_error.png')

desc_tdm['Top50'] = top50
min_error_c = error_path[error_path.argmin()]
logit_model_best = LogisticRegression(C = min_error_c, penalty = 'l1')
ally = desc_tdm.pop('Top50')
allX = desc_tdm
logit_fit_best = logit_model_best.fit(allX.values, ally.values)
keep_terms = desc_tdm.columns[np.where(logit_fit_best.coef_ > 0 )[1]]
print(Series(keep_terms))

trainX, testX, trainy, testy = cv.train_test_split(
    allX, ally, test_size = 0.2)
c_grid = [{'C': cs}]
n_cv_folds = 20

clf = GridSearchCV(LogisticRegression(C=1.0, penalty='l1'),  c_grid,
                   scoring=metrics.make_scorer(metrics.accuracy_score), cv=n_cv_folds)
clf.fit(trainX, trainy)

print(clf.best_params_, 1.0 - clf.best_score_)

rates = np.array([1.0 - x[1] for x in clf.grid_scores_])
stds = [np.std(1.0 - x[2])/math.sqrt(n_cv_folds) for x in clf.grid_scores_]
plt.figure(figsize=(8,8))
plt.fill_between(cs, rates - stds, rates + stds, color = 'steelblue', alpha = .4)
plt.plot(cs, rates, 'o-k', label = 'Avg. error rate across folds')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Avg. error rate (and +/- 1 s.e.)')
plt.legend(loc = 'best')
plt.gca().grid()
plt.savefig('GridCV.png')

print(metrics.classification_report(testy, clf.predict(testX)))
print(DataFrame(metrics.confusion_matrix(testy, clf.predict(testX))))