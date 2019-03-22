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


import sys 
sys.path.insert(0,'..')


# In[ ]:


from cnf_dataset import clauses_to_matrix
from dpll import DPLL, RandomClauseDPLL, MostCommonVarDPLL, RandomVarDPLL
from cnf import get_random_kcnf, CNF, get_sats_SR, get_pos_SR
from tqdm import tqdm
from collections import Counter


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


LIMIT_RUNS = 1000000


# In[ ]:


def shorten_cnf(cnf: CNF):
    for c in cnf.clauses:
        if len(c) == 1:
            # print("Chosen lone clause", c[0])
            return shorten_cnf(cnf.set_var(c[0]))
    all_literals = set(x
                       for clause in cnf.clauses
                       for x in clause)
    for v in cnf.vars:
        if v in all_literals and (-v) not in all_literals:
            # print("Chosen lone literal", v)
            return shorten_cnf(cnf.set_var(v))
        if (-v) in all_literals and v not in all_literals:
            # print("Chosen lone literal", -v)
            return shorten_cnf(cnf.set_var(-v))
    return cnf

def make_normalized(cls):
    class NormalizedDPLL(cls):
        def run(self, cnf: CNF):
            assert isinstance(cnf, CNF)
            self.number_of_runs += 1
            if self.number_of_runs > LIMIT_RUNS:
                return None
            
            cnf = shorten_cnf(cnf)
            if cnf.is_true():
                return []
            elif cnf.is_false():
                return None

            sug_var = self.suggest(cnf)
            sug_cnf = cnf.set_var(sug_var)

            sug_res = self.run(sug_cnf)
            if sug_res is not None:
                return [sug_var] + sug_res

            not_sug_cnf = cnf.set_var(-sug_var)
            not_sug_res = self.run(not_sug_cnf)
            if not_sug_res is not None:
                self.number_of_errors += 1
                return [-sug_var] + not_sug_res
            return None
    NormalizedDPLL.__name__ = "Normalized{}".format(cls.__name__)
    return NormalizedDPLL


# In[ ]:


np.set_printoptions(precision=3, suppress=True)


# In[ ]:


import tensorflow as tf
import os

BATCH_SIZE = 1


# In[ ]:


# Because we have to pass full batch

logfile = open("/tmp/log_2", "w")
#import sys
#logfile = sys.stdout

class GraphBasedDPLL(DPLL):
    def suggest(self, input_cnf: CNF):
        clause_num = len(input_cnf.clauses)
        var_num = max(input_cnf.vars)
        inputs = np.asarray([clauses_to_matrix(input_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
        
        policy_probs = predict_fn({"input": inputs})['policy_probabilities']
        
        print("CNF:", input_cnf, end=' ', file=logfile)
        print('cnf_clauses[0]', input_cnf.clauses, end=' ', file=logfile)
        print("probs:\n",  policy_probs[0], end='\n', file=logfile)
        
        best_prob = 0.0
        best_svar = None
        for var in input_cnf.vars:
            for svar in [var, -var]:
                svar_prob = policy_probs[0][var-1][0 if svar > 0 else 1]
                print(svar, svar_prob, best_prob, file=logfile)
                if svar_prob > best_prob:
                    best_prob = svar_prob
                    best_svar = svar
        print("best_svar:", best_svar, file=logfile)
        # print("Chosen neural", best_svar)
        return best_svar

class MostCommonDPLL(DPLL):
    def suggest(self, cnf: CNF):
        counter = Counter()
        for clause in cnf.clauses:
            for svar in clause:
                counter[svar] += 1
        return counter.most_common(1)[0][0]
    
class JeroslawDPLL(DPLL):
    def suggest(self, cnf: CNF):
        return jw(cnf.clauses)
    

class HumbleDPLL(DPLL):
    def suggest(self, input_cnf: CNF):
        clause_num = len(input_cnf.clauses)
        var_num = max(input_cnf.vars)
        inputs = np.asarray([clauses_to_matrix(input_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
        
        output = predict_fn({"input": inputs})
        policy_probs = output['policy_probabilities']
        sat_prob = output['sat_probabilities'][0]
        
        best_prob = 0.0
        best_svar = None
        for var in input_cnf.vars:
            for svar in [var, -var]:
                svar_prob = policy_probs[0][var-1][0 if svar > 0 else 1]
                #print(svar, svar_prob, best_prob, file=logfile)
                if svar_prob > best_prob:
                    best_prob = svar_prob
                    best_svar = svar
        if sat_prob < 0.3:
            # Overwriting with JW
            best_svar = jw(input_cnf.clauses)
        return best_svar    


# In[ ]:


NormalizedGraphBasedDPLL = make_normalized(GraphBasedDPLL)
NormalizedMostCommonDPLL = make_normalized(MostCommonDPLL)
NormalizedJeroslawDPLL = make_normalized(JeroslawDPLL)
NormalizedHumbleDPLL = make_normalized(HumbleDPLL)


# In[ ]:


class FastHumbleDPLL(DPLL):
    def run(self, cnf: CNF, fast=False):
        assert isinstance(cnf, CNF)
        self.number_of_runs += 1
        if self.number_of_runs > LIMIT_RUNS:
            return None

        cnf = shorten_cnf(cnf)
        if cnf.is_true():
            return []
        elif cnf.is_false():
            return None

        switch_to_heuristic, sug_var = self.suggest(cnf, fast)
        sug_cnf = cnf.set_var(sug_var)

        sug_res = self.run(sug_cnf, switch_to_heuristic)
        if sug_res is not None:
            return [sug_var] + sug_res

        not_sug_cnf = cnf.set_var(-sug_var)
        not_sug_res = self.run(not_sug_cnf, switch_to_heuristic)
        if not_sug_res is not None:
            self.number_of_errors += 1
            return [-sug_var] + not_sug_res
        return None

    def suggest(self, input_cnf: CNF, fast):
        if not fast:
            clause_num = len(input_cnf.clauses)
            var_num = max(input_cnf.vars)
            inputs = np.asarray([clauses_to_matrix(input_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)

            output = predict_fn({"input": inputs})
            policy_probs = output['policy_probabilities']
            sat_prob = output['sat_probabilities'][0]

            best_prob = 0.0
            best_svar = None
            for var in input_cnf.vars:
                for svar in [var, -var]:
                    svar_prob = policy_probs[0][var-1][0 if svar > 0 else 1]
                    #print(svar, svar_prob, best_prob, file=logfile)
                    if svar_prob > best_prob:
                        best_prob = svar_prob
                        best_svar = svar
        if fast or sat_prob < 0.3:
            # Overwriting with JW
            best_svar = jw(input_cnf.clauses)
        return (fast or sat_prob < 0.3), best_svar    


# In[ ]:


def compute_steps(sats, dpll_cls):
    steps = []
    errors = []
    solved = 0
    for sat in tqdm(sats):
        dpll = dpll_cls()
        res = dpll.run(sat)
        # assert res is not None
        if res is not None:
            steps.append(dpll.number_of_runs)
            errors.append(dpll.number_of_errors)
            solved += 1
    print("Within {} steps solved {} problems out of {}".format(LIMIT_RUNS, solved, len(sats)))
    return steps, errors


# In[ ]:


def compute_and_print_steps(sats, dpll_cls):
    print(dpll_cls.__name__)
    steps, errors = compute_steps(sats, dpll_cls)
    print("#Sats: {}; avg step: {:.2f}; stdev step: {:.2f}; avg error: {:.2f}; stdev error: {:.2f}".format(
        len(steps), np.mean(steps), np.std(steps), np.mean(errors), np.std(errors)))
    print("Table: ", steps)


# In[ ]:


def print_all(s, n, m, light=False):
    # def get_sats_SR(sample_number, min_variable_number, clause_number, max_variable_number=None):
    global S, N, M
    S = s
    N = n # number of clauses
    M = m # number of variables
    
    MAX_TRIES = 100000
    sats = []
    
    random.seed(1)
    np.random.seed(1)
    
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
    compute_and_print_steps(sats, FastHumbleDPLL)
    # if not light:
    #     compute_and_print_steps(sats, NormalizedMostCommonDPLL)
    compute_and_print_steps(sats, NormalizedJeroslawDPLL)
    # compute_and_print_steps(sats, NormalizedHumbleDPLL)
    # if not light:
    #     compute_and_print_steps(sats, NormalizedGraphBasedDPLL)

    logfile.flush()


# In[ ]:


print_all(100, 10000, 60, light=True)


# In[ ]:


print_all(100, 10000, 70, light=True)


# In[ ]:


print_all(100, 10000, 80, light=True)


# In[ ]:


print_all(100, 10000, 90, light=True)


# In[ ]:


print_all(100, 10000, 100, light=True)


# In[ ]:


print_all(100, 10000, 110, light=True)


# In[ ]:


print_all(100, 10000, 120, light=True)


# In[ ]:


print_all(100, 10000, 130, light=True)

