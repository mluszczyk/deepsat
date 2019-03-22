#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.contrib import predictor

export_dir = "gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-02/1552773431/"
predict_fn = predictor.from_saved_model(export_dir)


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import random


# In[ ]:


import tqdm


# In[ ]:


from time import time


# In[ ]:


import sys 
sys.path.insert(0,'../..')


# In[ ]:


from cnf_dataset import clauses_to_matrix
from dpll import DPLL, RandomClauseDPLL, MostCommonVarDPLL, RandomVarDPLL, JeroslowWangDPLL
import cdcl
CDCL = cdcl.CDCL
from cnf import get_random_kcnf, CNF, get_sats_SR, get_pos_SR, get_sat_SR, get_random_sat_kcnf
from tqdm import tqdm
from collections import Counter


# In[ ]:


LIMIT_RUNS = 1000


# In[ ]:


CDCL().run(CNF([[-1, -2], [+1, -2], [+1, +2]]))


# In[ ]:


JeroslowWangDPLL().run(CNF([[-1, -2], [+1, -2], [+1, +2]]))


# In[ ]:


cnf = get_pos_SR(15, 15, 2000)
print("Generated")
# print(cnf)
start = time()
CDCL().run(cnf)
end = time()
print(end-start)
start = time()
DPLL().run(cnf)
end = time()
print(end-start)


# In[ ]:


import math
from collections import defaultdict

def jw(clauses):
    score = defaultdict(int)

    for clause in clauses:
        for l in clause:
            score[l] += 2. ** (-len(clause))

    return max(score, key=score.get)


# In[ ]:


jw([[-1, -2], [+1, -2]])


# In[ ]:


np.set_printoptions(precision=3, suppress=True)


# In[ ]:


import tensorflow as tf
import os

BATCH_SIZE = 1


# In[ ]:


VERBOSE = False
TIMEOUT = 10000
cdcl.VERBOSE = VERBOSE
cdcl.TIMEOUT = TIMEOUT

class GraphBasedCDCL(CDCL):
    def suggest(self, input_cnf: CNF):
        clause_num = len(input_cnf.clauses)
        var_num = max(input_cnf.vars)
        inputs = np.asarray([clauses_to_matrix(input_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
        
        output = predict_fn({"input": inputs})
        policy_probs = output['policy_probabilities']
        sat_prob = output['sat_probabilities'][0]
        
        #print("CNF:", input_cnf, end=' ', file=logfile)
        #print('cnf_clauses[0]', input_cnf.clauses, end=' ', file=logfile)
        #print("probs:\n",  policy_probs[0], end='\n', file=logfile)
        
        best_prob = 0.0
        best_svar = None
        for var in input_cnf.vars:
            for svar in [var, -var]:
                svar_prob = policy_probs[0][var-1][0 if svar > 0 else 1]
                #print(svar, svar_prob, best_prob, file=logfile)
                if svar_prob > best_prob:
                    best_prob = svar_prob
                    best_svar = svar
        #print("best_svar:", best_svar, file=logfile)
        if VERBOSE:
            print("Chosen neural", best_svar)
            print("Pred SAT prob", sat_prob)
        return best_svar

class HumbleGraphBasedCDCL(CDCL):
    def suggest(self, input_cnf: CNF):
        clause_num = len(input_cnf.clauses)
        var_num = max(input_cnf.vars)
        inputs = np.asarray([clauses_to_matrix(input_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
        
        output = predict_fn({"input": inputs})
        policy_probs = output['policy_probabilities']
        sat_prob = output['sat_probabilities'][0]
        
        #print("CNF:", input_cnf, end=' ', file=logfile)
        #print('cnf_clauses[0]', input_cnf.clauses, end=' ', file=logfile)
        #print("probs:\n",  policy_probs[0], end='\n', file=logfile)
        
        best_prob = 0.0
        best_svar = None
        for var in input_cnf.vars:
            for svar in [var, -var]:
                svar_prob = policy_probs[0][var-1][0 if svar > 0 else 1]
                #print(svar, svar_prob, best_prob, file=logfile)
                if svar_prob > best_prob:
                    best_prob = svar_prob
                    best_svar = svar
        #print("best_svar:", best_svar, file=logfile)
        if VERBOSE:
            print("Chosen neural", best_svar)
            print("Pred SAT prob", sat_prob)
        if sat_prob < 0.3:
            # Overwriting with JW
            best_svar = jw(input_cnf.clauses)
            if VERBOSE:
                print("Choosing JW", best_svar)
        #if sat_prob < 0.02:
        #    print("Aborting, prob too small")
        #    raise ValueError("Prob too small")
        return best_svar

class JeroslawCDCL(CDCL):
    def suggest(self, cnf: CNF):
        res = jw(cnf.clauses)
        if VERBOSE:
            print("Chosen JW", res)
        return res


# In[ ]:


HumbleGraphBasedCDCL().run(CNF([[-1, -2], [+1, +3], [+2, -3], [+3, -2], [+1, -2], [2, 3]]))


# In[ ]:


def compute_steps(sats, cdcl_cls):
    steps = []
    solved = 0
    for sat in tqdm.tqdm(sats):
        cdcl = cdcl_cls()
        res = cdcl.run(sat)
        # assert res is not None
        if res is not None:
            steps.append(cdcl.number_of_runs)
            solved += 1
    print("Within {} steps solved {} problems out of {}".format(LIMIT_RUNS, solved, len(sats)))
    return steps


# In[ ]:


def compute_and_print_steps(sats, dpll_cls):
    try:
        print("")
        print("Results of {}".format(dpll_cls.__name__))
        steps = compute_steps(sats, dpll_cls)
        print("#Sats: {}; avg step: {:.2f}; stdev step: {:.2f}".format(
            len(steps), np.mean(steps), np.std(steps)))
        print("Table: {}".format(steps))
    except Exception as e:
        print(e)
        print("Timeout!", TIMEOUT, "steps")


# In[ ]:


def print_all(s, n, m, light=False, seed=1, to_test=[HumbleGraphBasedCDCL, JeroslawCDCL]):
    print("Starting...")
    # def get_sats_SR(sample_number, min_variable_number, clause_number, max_variable_number=None):
    global S, N, M
    S = s
    N = n # number of clauses
    M = m # number of variables
    
    MAX_TRIES = 100000
    sats = []
    
    random.seed(seed)
    np.random.seed(seed)
    
    for index in range(MAX_TRIES):
        if len(sats) >= S:
            break
        sat = get_pos_SR(M, M, N)
        # if DPLL().run(sat) is not None:
        sats.append(sat)
    assert len(sats) == S
    # sats = get_sats_SR(S,M,N)
    # for sat in sats:
    #    print(sat)
    # assert len(sats) == S
    print("We have generated {} formulas".format(len(sats)))
    # compute_and_print_steps(sats, DPLL)
    # compute_and_print_steps(sats, SimplifiedGraphBasedDPLL)
    
    
    # compute_and_print_steps(sats, SimplifiedMostCommonDPLL)
    # compute_and_print_steps(sats, NormalizedMostCommonDPLL)
    
    #compute_and_print_steps(sats, GraphBasedCDCL)
    for method in to_test:
        compute_and_print_steps(sats, method)
    #logfile.flush()


# In[ ]:


#print_all(10, 200, 50)


# In[ ]:


print_all(100, 1000, 30)


# In[ ]:


print_all(100, 1000, 70)


# In[ ]:


print_all(100, 2000, 80)


# In[ ]:


print_all(100, 2000, 90)


# In[ ]:


print_all(100, 2000, 100)


# In[ ]:


print_all(110, 2000, 110)


# In[ ]:




