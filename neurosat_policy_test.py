import tensorflow as tf

import neurosat_policy


class TestNeurosatPolicy(tf.test.TestCase):
    @tf.test.mock.patch.dict(neurosat_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                                "SAMPLES": 64})
    def test_train_sr(self):
        self.assertTrue(neurosat_policy.DEFAULT_SETTINGS["SR_GENERATOR"])
        neurosat_policy.main()

    @tf.test.mock.patch.dict(neurosat_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                                "SAMPLES": 64,
                                                                "SR_GENERATOR": False})
    def test_train_not_sr(self):
        neurosat_policy.main()