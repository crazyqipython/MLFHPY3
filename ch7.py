import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import scipy.optimize as opt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import random
import math
from data import data_dir

file_path = os.path.join(data_dir, "01_heights_weights_genders.csv")
file_path1 = os.path.join(data_dir, "lexical_database.csv")
heights_weights = read_csv(file_path)

# ols_fit = ols('Weight ~Height', data = heights_weights).fit()
# ols_sse = ols_fit.mse_resid *(ols_fit.df_resid)
# print(np.round(ols_fit.params,3))
# print("MSE %i" %round(ols_sse))
#
# y = heights_weights['Weight'].values
# Xmat = sm.add_constant(heights_weights['Height'], prepend=True)
#
# def ridge_error(params, y, Xmat, lam):
#     predicted = np.dot(Xmat, params)
#     sse = ((y-predicted)**2).sum()
#     sse += lam*(params**2).sum()
#     return sse
#
# def ridge_grad(params, y, Xmat, lam):
#     grad = np.dot(np.dot(Xmat.T, Xmat), params) - np.dot(Xmat.T, y)
#     grad += lam**params
#     grad *=2
#     return grad
#
# def ridge_hess(params, y, Xmat, lam):
#     return np.dot(Xmat.T, Xmat) + np.eye(2) * lam
#
# print('&&&&&&&&&&&&&&&&&&')
# params0 = np.array([0.0, 0.0])
# ridge_fit = opt.fmin(ridge_error,  params0, args=(y, Xmat, 1))
# print('Solution: a = %8.3f, b = %8.3f ' % tuple(ridge_fit))
#
# # With Newton conjugate gradient
# print('\nNewton conjugate gradient without hession')
# ridge_fit = opt.fmin_ncg(ridge_error, params0, fprime=ridge_grad, args=(y, Xmat, 1))
# print('Solution: a = %8.3f, b = %8.3f ' % tuple(ridge_fit))
#
# print('\n Newton with the hession')
# ridge_fit = opt.fmin_ncg(ridge_error, params0, fprime=ridge_grad, fhess=ridge_hess, args=(y, Xmat,1))
# print('Solution: a = %8.3f, b = %8.3f ' % tuple(ridge_fit))
#
# print('\nUse BFGS')
# ridge_fit = opt.fmin_bfgs(ridge_error, params0, fprime=ridge_grad, args=(y, Xmat, 1))
# print('Solution: ', ridge_fit)



"""
Deciphering text with the Metropolis-Hastings Algorithm
"""
lexical_database = read_csv(file_path1, index_col=0, header =None, skiprows=1, squeeze=True)
lexical_database.index.name = 'word'

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
           'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
           'y', 'z']
ceasar_cipher = {i:j for i,j in zip(letters, letters[1:] + letters[:1])}
inverse_ceasar_cipher = {ceasar_cipher[k]:k for k in ceasar_cipher}

def cipher_text(text, cipher_dict = ceasar_cipher):
    strlist = list(text)
    ciphered = ''.join(cipher_dict.get(x) or x for x in strlist)
    return ciphered

def decipher_text(text, cipher_dict = ceasar_cipher):
    strlist = list(text)
    decipher_dict = {cipher_dict[k]: k for k in cipher_dict}
    deciphered = ''.join([decipher_dict.get(x) or x for x in strlist])
    return deciphered

def generate_random_cipher():
    cipher = []
    input = letters
    output = letters[:]
    random.shuffle(output)
    cipher_dict = {k:v for (k, v) in zip(input, output)}
    return cipher_dict

def modify_cipher(cipher_dict, input, new_output):
    decipher_dict = {cipher_dict[k]: k for k in cipher_dict}
    old_output = cipher_dict[input]

    new_cipher = cipher_dict.copy()
    new_cipher[input] = new_output
    new_cipher[decipher_dict[new_output]] = old_output
    return new_cipher

def propose_modified_cipher_from_cipher(text, cipher_dict,
                                        lexical_db = lexical_database):
    _ = text
    input = random.sample(cipher_dict.keys(), 1)[0]
    new_output = random.sample(letters, 1)[0]
    return modify_cipher(cipher_dict, input, new_output)

def propose_modified_cipher_from_text(text, cipher_dict,
                                      lexical_db = lexical_database):
    deciphered = decipher_text(text, cipher_dict).split()
    letters_to_sample = ''.join([t for t in deciphered if lexical_db.get(t) is None])
    letters_to_sample = letters_to_sample or ''.join(set(deciphered))
    input = random.sample(letters_to_sample, 1)[0]
    new_output = random.sample(letters, 1)[0]
    return modify_cipher(cipher_dict, input, new_output)


def one_gram_prob(one_gram, lexical_db=lexical_database):
    return lexical_db.get(one_gram) or np.finfo(float).eps

def text_logp(text, cipher_dict, lexical_db = lexical_database):
    deciphered = decipher_text(text, cipher_dict).split()
    logp = np.array([math.log(one_gram_prob(w)) for w in deciphered]).sum()
    return logp

def metropolis_step(text, cipher_dict, proposal_rule, lexical_db = lexical_database):
    proposed_cipher = proposal_rule(text, cipher_dict)
    lp1 = text_logp(text, cipher_dict)
    lp2 = text_logp(text, proposed_cipher)
    if lp2 > lp1:
        return proposed_cipher
    else:
        a = math.exp(lp2 - lp1)
        x = random.random()
        if x < a:
            return proposed_cipher
        else:
            return cipher_dict

message = 'here is some sample text'
ciphered_text = cipher_text(message, ceasar_cipher)
niter = 100

def metropolis_decipher(ciphered_text,  proposal_rule, niter, seed = 4):
    random.seed(seed)
    cipher = generate_random_cipher()
    deciphered_text_list = []

    logp_list = []

    for i in range(niter):
        print(i)
        logp = text_logp(ciphered_text, cipher)
        current_deciphered_text = decipher_text(ciphered_text, cipher)
        deciphered_text_list.append(current_deciphered_text)
        logp_list.append(logp)

        cipher = metropolis_step(ciphered_text, cipher, proposal_rule)

    results = DataFrame({'deciphered_text': deciphered_text_list, 'logp': logp_list})
    results.index = np.arange(1, niter+1)
    return results

results0 = metropolis_decipher(ciphered_text,
                               propose_modified_cipher_from_cipher, niter)

a = results0.ix[10::10]
print(a)

results1 = metropolis_decipher(ciphered_text,
                               propose_modified_cipher_from_text, niter)
a = results1.ix[10::10]
print(a)