import tensorflow as tf

import lstm_policy


class TestGraphPolicy(tf.test.TestCase):
    @tf.test.mock.patch.dict(lstm_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                            "SAMPLES": 64})
    def test_train_not_sr(self):
        self.assertFalse(lstm_policy.DEFAULT_SETTINGS["SR_GENERATOR"])
        self.assertEqual(lstm_policy.DEFAULT_SETTINGS["CLAUSE_AGGREGATION"], "LSTM")
        lstm_policy.main()

    @tf.test.mock.patch.dict(lstm_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                            "SAMPLES": 64,
                                                            "CLAUSE_AGGREGATION": "BOW"})
    def test_train_not_sr_bow(self):
        self.assertFalse(lstm_policy.DEFAULT_SETTINGS["SR_GENERATOR"])
        lstm_policy.main()

    @tf.test.mock.patch.dict(lstm_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                            "SAMPLES": 64,
                                                            "SR_GENERATOR": True,
                                                            "CLAUSE_SIZE": 8})
    def test_train_sr(self):
        lstm_policy.main()
