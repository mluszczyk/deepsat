import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
import tensorflow as tf
import os
from cnf_dataset import pad_and_concat
from dpll import DPLL, RandomClauseDPLL, MostCommonVarDPLL, RandomVarDPLL
from cnf import get_random_kcnf, CNF
from collections import namedtuple


GraphEnv = namedtuple(
    "GraphEnv",
    ["sess", "batch_size", "clause_size",
     "g_policy_probs", "g_inputs", "g_lengths", "g_sat_probs"])


def default_experiments(meta_dir, expected_variable_num):
    np.set_printoptions(precision=3, suppress=True)

    experiments = [
        (100, 2, 2, 3),
        (100, 2, 4, 20)
    ]
    for var_num in range(4, expected_variable_num + 1):
        experiments += [(100, 3, var_num, var_num * 5),
                        (100, 3, var_num, var_num * 10)]

    sess = tf.Session()
    graph_env = load_graph_env(sess, meta_dir, expected_variable_num)
    test_graph(graph_env)
    lstm_dpll_cls = make_lstm_dpll_class(graph_env)
    return execute_experiments(experiments, [lstm_dpll_cls])


def load_graph_env(sess, meta_dir, expected_variable_num):
    batch_size = 128

    meta_file = get_most_fresh_meta(os.path.expandvars(meta_dir))
    clause_size = 3

    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_file)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta_file)))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()

    g_inputs = graph.get_tensor_by_name("inputs:0")
    g_lengths = graph.get_tensor_by_name("lengths:0")

    g_policy_probs = graph.get_tensor_by_name('policy_prob:0')
    g_sat_probs = graph.get_tensor_by_name('sat_prob:0')

    assert g_policy_probs.shape[1] == 2 * expected_variable_num

    return GraphEnv(sess=sess,
                    batch_size=batch_size,
                    clause_size=clause_size,
                    g_policy_probs=g_policy_probs,
                    g_inputs=g_inputs,
                    g_lengths=g_lengths,
                    g_sat_probs=g_sat_probs)


def test_graph(graph_env):
    inputs = np.asarray([[[1, 1, 3], [1, -3, 2], [1, 3, 4], [-2, -3, 4]]] * graph_env.batch_size, dtype=np.int32)
    lengths = np.asarray([inputs.shape[1]] * graph_env.batch_size, dtype=np.int32)

    sat_prob, policy_probs = graph_env.sess.run(
        [graph_env.g_sat_probs, graph_env.g_policy_probs],
        feed_dict={graph_env.g_inputs: inputs, graph_env.g_lengths: lengths})

    print(sat_prob[0], policy_probs[0])


def get_most_fresh_meta(path):
    meta_files = [item for item in os.listdir(path) if item.endswith(".meta")]
    meta_files.sort()
    return os.path.join(path, meta_files[-1])


class LSTMBasedDPLLTemplate(DPLL):
    def __init__(self, graph_env):
        super().__init__()
        self.sess = graph_env.sess
        self.batch_size = graph_env.batch_size
        self.clause_size = graph_env.clause_size
        self.g_policy_probs = graph_env.g_policy_probs
        self.g_inputs = graph_env.g_inputs
        self.g_lengths = graph_env.g_lengths

    def suggest(self, input_cnf: CNF):
        cnfs = [input_cnf] * self.batch_size
        cnfs_clauses = [[claus + tuple([claus[0]] * (3 - len(claus))) for claus in cnf.clauses] for cnf in cnfs]
        inputs, lengths = pad_and_concat(cnfs_clauses, self.clause_size)

        policy_probs = self.sess.run(self.g_policy_probs, feed_dict={self.g_inputs: inputs, self.g_lengths: lengths})

        best_prob = 0.0
        best_svar = None
        for var in input_cnf.vars:
            for svar in [var, -var]:
                svar_prob = policy_probs[0][(var - 1) * 2 + (0 if svar > 0 else 1)]
                if svar_prob > best_prob:
                    best_prob = svar_prob
                    best_svar = svar
        return best_svar


def make_lstm_dpll_class(graph_env):
    """This is a workaround, because DPLL code expects that __init__ does not take any arguments."""
    class LSTMBasedDPLL(LSTMBasedDPLLTemplate):
        def __init__(self):
            super().__init__(graph_env)

    return LSTMBasedDPLL


def compute_steps(sats, dpll_cls):
    steps = []
    errors = []
    for sat in sats:
        dpll = dpll_cls()
        res = dpll.run(sat)
        assert res is not None
        steps.append(dpll.number_of_runs)
        errors.append(dpll.number_of_errors)
    return steps, errors


def compute_and_print_steps(sats, dpll_cls):
    steps, errors = compute_steps(sats, dpll_cls)
    print("#Sats: {}; avg step: {:.2f}; stdev step: {:.2f}; avg error: {:.2f}; stdev error: {:.2f}".format(
        len(steps), np.mean(steps), np.std(steps), np.mean(errors), np.std(errors)))
    stats = {
        "steps": len(steps),
        "avg_step": np.mean(steps),
        "std_step": np.std(steps),
        "avg_error": np.mean(errors),
        "std_error": np.std(errors)
    }
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.title("Steps of {}".format(dpll_cls.__name__))
    plt.hist(steps, bins=range(2 ** (N + 1)))
    plt.ylim((0, len(sats)))

    plt.subplot(1, 2, 2)
    plt.title("Errors of {}".format(dpll_cls.__name__))
    plt.hist(errors, bins=range(N + 1))
    plt.ylim((0, len(sats)))
    plt.show()
    return stats


def print_all(s, k, n, m, all_stats, dpll_classes):
    global S, K, N, M
    S = s
    K = k
    N = n
    M = m

    MAX_TRIES = 100000
    sats = []
    for index in range(MAX_TRIES):
        if len(sats) >= S:
            break
        sat = get_random_kcnf(K, N, M)
        if DPLL().run(sat) is not None:
            sats.append(sat)
    assert len(sats) == S
    for dpll_cls in [DPLL, RandomVarDPLL, RandomClauseDPLL, MostCommonVarDPLL] + dpll_classes:
        stats = compute_and_print_steps(sats, dpll_cls)
        stats.update({"dpll_type": dpll_cls.__name__, "s": S, "k": k, "n": n, "m": m})
        all_stats.append(stats)


def execute_experiments(experiments, dpll_classes):
    all_stats = []
    for experiment in experiments:
        print("S: {} K: {} N: {} M: {}".format(*experiment))
        print_all(*experiment, all_stats, dpll_classes)
    return all_stats


def summary_table_and_plots(data):
    df = pd.DataFrame(data)

    df["randomvar_avg_error"] = np.where(df.dpll_type == 'RandomVarDPLL', df.avg_error, None)
    df["mostcommon_avg_error"] = np.where(df.dpll_type == 'MostCommonVarDPLL', df.avg_error, None)
    df["lstm_avg_error"] = np.where(df.dpll_type == 'LSTMBasedDPLL', df.avg_error, None)

    df["randomvar_avg_step"] = np.where(df.dpll_type == 'RandomVarDPLL', df.avg_step, None)
    df["mostcommon_avg_step"] = np.where(df.dpll_type == 'MostCommonVarDPLL', df.avg_step, None)
    df["lstm_avg_step"] = np.where(df.dpll_type == 'LSTMBasedDPLL', df.avg_step, None)

    del df["steps"], df["std_step"], df["std_error"], df["s"], df["dpll_type"], df["avg_step"], df["avg_error"]
    df = df.groupby(["k", "m", "n"]).first()
    display(df)

    ylabels = list(range(df.shape[0]))
    plt.title("average step number")
    plt.plot(ylabels, df.randomvar_avg_step, label='randomvar')
    plt.plot(ylabels, df.mostcommon_avg_step, label='mostcommon')
    plt.plot(ylabels, df.lstm_avg_step, label='lstm')
    plt.legend()
    plt.show()

    plt.title("average error number")
    plt.plot(ylabels, df.randomvar_avg_error, label='randomvar')
    plt.plot(ylabels, df.mostcommon_avg_error, label='mostcommon')
    plt.plot(ylabels, df.lstm_avg_error, label='lstm')
    plt.legend()
    plt.show()
