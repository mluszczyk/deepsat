import numpy as np

import multiprocessing

from cnf import get_sats_SR, get_random_kcnfs
from timed import timed


@timed
def gen_labels(variable_num, pool, cnfs):
    sat_labels = pool.map(satisfiable, cnfs)

    sats_to_check = [(cnf, is_satisfiable, literal)
                     for (cnf, is_satisfiable) in zip(cnfs, sat_labels)
                     for v in range(1, variable_num + 1)
                     for literal in [v, -v]]
    policy_labels = np.asarray(pool.map(set_and_sat, sats_to_check))
    policy_labels = np.asarray(policy_labels).reshape(len(cnfs), variable_num, 2)
    assert len(cnfs) == len(sat_labels) == policy_labels.shape[0]
    assert policy_labels.shape[1] == variable_num
    assert policy_labels.shape[2] == 2

    return sat_labels, policy_labels


@timed
def gen_cnfs_with_labels(options, pool):
    if options['SR_GENERATOR']:
        cnfs = get_sats_SR(options['BATCH_SIZE'], options['MIN_VARIABLE_NUM'], options['CLAUSE_NUM'],
                           options['VARIABLE_NUM'])
    else:
        cnfs = get_random_kcnfs(options['BATCH_SIZE'], options['CLAUSE_SIZE'], options['VARIABLE_NUM'],
                                options['CLAUSE_NUM'],
                                min_clause_number=options['MIN_CLAUSE_NUM'])
    sat_labels, policy_labels = gen_labels(options['VARIABLE_NUM'], pool, cnfs)
    return cnfs, sat_labels, policy_labels


class PoolDatasetGenerator:
    def __init__(self, options):
        self.options = options
        self.pool = None

    def __enter__(self):
        self.pool = multiprocessing.Pool(self.options['PROCESSOR_NUM']).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.__exit__(exc_type, exc_val, exc_tb)

    def generate_batch(self):
        cnfs, sat_labels, policy_labels = gen_cnfs_with_labels(self.options, self.pool)
        inputs = np.asarray([clauses_to_matrix(cnf.clauses, self.options['CLAUSE_NUM'], self.options['VARIABLE_NUM']) for cnf in cnfs])
        return inputs, sat_labels, policy_labels


def satisfiable(cnf):
    return cnf.satisfiable()


def set_and_sat(triple):
    cnf, is_satisfiable, literal = triple
    if not is_satisfiable or not abs(literal) in cnf.vars:
        return False
    cnf = cnf.set_var(literal)
    return cnf.satisfiable()


def clauses_to_matrix(clauses, max_clause_num, max_variable_num):
    def var_in_clause_val(var, i):
        if i >= len(clauses):
            return [0.0, 0.0]
        return [1.0 if var in clauses[i] else 0.0,
                1.0 if -var in clauses[i] else 0.0]

    result = [[var_in_clause_val(var, i) for i in range(max_clause_num)]
              for var in range(1, max_variable_num + 1)]
    return result
