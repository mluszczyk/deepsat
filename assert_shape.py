def assert_shape(matrix, shape: list):
    act_shape = matrix.get_shape().as_list()
    assert act_shape == shape, "got shape {}, expected {}".format(act_shape, shape)