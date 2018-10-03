import unittest

import numpy as np

import cnf_dataset


class OnlineDatasetCase(unittest.TestCase):
    def test_pool_generator(self):
        options = {
            "VARIABLE_NUM": 5,
            "CLAUSE_NUM": 5,
            "SR_GENERATOR": False,
            "PROCESSOR_NUM": 1,
            "MIN_CLAUSE_NUM": 2,
            "CLAUSE_SIZE": 5,
            "BATCH_SIZE": 5,
            # "MIN_VARIABLE_NUM": 1,  # only for SR
        }
        with cnf_dataset.PoolDatasetGenerator(options) as dataset_generator:
            cnfs_vals, sat_labels_vals, policy_labels_vals = dataset_generator.generate_batch()

        self.assertEqual(cnfs_vals.shape, (options["BATCH_SIZE"], options["CLAUSE_NUM"], options["VARIABLE_NUM"], 2))
        self.assertEqual(np.asarray(sat_labels_vals).shape, (options["BATCH_SIZE"],))
        self.assertEqual(np.asarray(policy_labels_vals).shape, (options["BATCH_SIZE"], options["VARIABLE_NUM"], 2))


if __name__ == '__main__':
    unittest.main()
