import tensorflow as tf

import neurosat_policy


class TestNeurosatPolicy(tf.test.TestCase):
    @tf.test.mock.patch("neurosat_policy.NEPTUNE_ENABLED", False)
    @tf.test.mock.patch("neurosat_policy.SAMPLES", 64)
    def test_train_sr(self):
        self.assertTrue(neurosat_policy.SR_GENERATOR)
        neurosat_policy.main()

    @tf.test.mock.patch("neurosat_policy.NEPTUNE_ENABLED", False)
    @tf.test.mock.patch("neurosat_policy.SAMPLES", 64)
    @tf.test.mock.patch("neurosat_policy.SR_GENERATOR", False)
    def test_train_not_sr(self):
        neurosat_policy.main()