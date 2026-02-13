def test_pymesh():
    import numpy as np
    import pymesh

    # pymesh.test()
    # due to the deprecated function of distutils.ccompiler
    # we cannot run the test function

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    edges = np.array([[0, 1], [1, 3], [2, 3], [2, 0]])
    wire_network = pymesh.wires.WireNetwork.create_from_data(vertices, edges)

    assert wire_network.num_vertices == 4
    assert wire_network.dim == 3
    assert wire_network.num_edges == 4
