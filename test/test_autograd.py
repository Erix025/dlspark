import sys

sys.path.append("./python")
sys.path.append("./apps")

import numpy as np
import dlspark
import torch

### Test for forward pass of all functions
def test_power_scalar_forward():
    value = np.random.randn(5, 4)
    np.testing.assert_allclose(
        dlspark.power_scalar(dlspark.Tensor(value), scalar=2).data,
        np.power(np.array(value), 2),
    )


def test_ewisepow_forward():
    a = np.random.randn(5, 4)
    b = np.array([[0,0,0,3]])
    np.testing.assert_allclose(
        dlspark.power(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.pow(a, b),
    )


def test_divide_forward():
    a = np.random.randn(3, 5)
    b = np.random.randn(3, 5)
    np.testing.assert_allclose(
        dlspark.divide(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.divide(a, b),
    )


def test_divide_scalar_forward():
    a = np.random.randn(10, 4)
    scalar = 100 * (np.random.rand() + 0.1)
    np.testing.assert_allclose(
        dlspark.divide_scalar(dlspark.Tensor(a), scalar=scalar).data,
        np.divide(a, scalar),
    )


def test_matmul_forward():
    # Test for nxn matrices
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 3)
    np.testing.assert_allclose(
        dlspark.matmul(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.matmul(a, b),
    )
    # Test for nxm and mxp matrices
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    np.testing.assert_allclose(
        dlspark.matmul(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.matmul(a, b),
    )
    # Test for 3D tensor
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 4, 5)
    np.testing.assert_allclose(
        dlspark.matmul(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.matmul(a, b),
    )
    # Test for 3D tensor and 2D tensor
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4, 5)
    np.testing.assert_allclose(
        dlspark.matmul(
            dlspark.Tensor(a),
            dlspark.Tensor(b),
        ).data,
        np.matmul(a, b),
    )


def test_summation_forward():
    a = np.random.randn(5, 4)
    np.testing.assert_allclose(
        dlspark.summation(
            dlspark.Tensor(a),
            axes=None,
        ).data,
        np.sum(a),
    )
    np.testing.assert_allclose(
        dlspark.summation(
            dlspark.Tensor(a),
            axes=1,
        ).data,
        np.sum(a, axis=1),
    )
    np.testing.assert_allclose(
        dlspark.summation(
            dlspark.Tensor(a),
            axes=0,
        ).data,
        np.sum(a, axis=0),
    )


def test_broadcast_to_forward():
    a = np.random.randn(3, 1)
    np.testing.assert_allclose(
        dlspark.broadcast_to(dlspark.Tensor(a), shape=(3, 3, 3)).data,
        np.broadcast_to(a, (3, 3, 3)),
    )


def test_reshape_forward():
    # Test for 2D tensor to 1D tensor
    a = np.random.randn(5, 3)
    np.testing.assert_allclose(
        dlspark.reshape(
            dlspark.Tensor(a),
            shape=(15,),
        ).data,
        np.reshape(a, (15,)),
    )
    # Test for 3D tensor
    a = np.random.randn(3, 2, 4)
    np.testing.assert_allclose(
        dlspark.reshape(
            dlspark.Tensor(a),
            shape=(2, 3, 4),
        ).data,
        np.reshape(a, (2, 3, 4)),
    )


def test_negate_forward():
    a = np.random.randn(5, 4)
    np.testing.assert_allclose(
        dlspark.negate(dlspark.Tensor(a)).data, np.negative(a)
    )


def test_transpose_forward():
    np.testing.assert_allclose(
        dlspark.transpose(dlspark.Tensor([[[1.95]], [[2.7]], [[3.75]]]), axes=(1, 2)).data,
        np.array([[[1.95]], [[2.7]], [[3.75]]]),
    )
    np.testing.assert_allclose(
        dlspark.transpose(
            dlspark.Tensor([[[[0.95]]], [[[2.55]]], [[[0.45]]]]), axes=(2, 3)
        ).data,
        np.array([[[[0.95]]], [[[2.55]]], [[[0.45]]]]),
    )
    np.testing.assert_allclose(
        dlspark.transpose(
            dlspark.Tensor(
                [
                    [[[0.4, 0.05], [2.95, 1.3]], [[4.8, 1.2], [1.65, 3.1]]],
                    [[[1.45, 3.05], [2.25, 0.1]], [[0.45, 4.75], [1.5, 1.8]]],
                    [[[1.5, 4.65], [1.35, 2.7]], [[2.0, 1.65], [2.05, 1.2]]],
                ]
            )
        ).data,
        np.array(
            [
                [[[0.4, 2.95], [0.05, 1.3]], [[4.8, 1.65], [1.2, 3.1]]],
                [[[1.45, 2.25], [3.05, 0.1]], [[0.45, 1.5], [4.75, 1.8]]],
                [[[1.5, 1.35], [4.65, 2.7]], [[2.0, 2.05], [1.65, 1.2]]],
            ]
        ),
    )
    np.testing.assert_allclose(
        dlspark.transpose(dlspark.Tensor([[[2.45]], [[3.5]], [[0.9]]]), axes=(0, 1)).data,
        np.array([[[2.45], [3.5], [0.9]]]),
    )
    np.testing.assert_allclose(
        dlspark.transpose(dlspark.Tensor([[4.4, 2.05], [1.85, 2.25], [0.15, 1.4]])).data,
        np.array([[4.4, 1.85, 0.15], [2.05, 2.25, 1.4]]),
    )
    np.testing.assert_allclose(
        dlspark.transpose(
            dlspark.Tensor([[0.05, 3.7, 1.35], [4.45, 3.25, 1.95], [2.45, 4.4, 4.5]])
        ).data,
        np.array([[0.05, 4.45, 2.45], [3.7, 3.25, 4.4], [1.35, 1.95, 4.5]]),
    )
    np.testing.assert_allclose(
        dlspark.transpose(
            dlspark.Tensor(
                [
                    [[0.55, 1.8, 0.2], [0.8, 2.75, 3.7], [0.95, 1.4, 0.8]],
                    [[0.75, 1.6, 1.35], [3.75, 4.0, 4.55], [1.85, 2.5, 4.8]],
                    [[0.2, 3.35, 3.4], [0.3, 4.85, 4.85], [4.35, 4.25, 3.05]],
                ]
            ),
            axes=(0, 1),
        ).data,
        np.array(
            [
                [[0.55, 1.8, 0.2], [0.75, 1.6, 1.35], [0.2, 3.35, 3.4]],
                [[0.8, 2.75, 3.7], [3.75, 4.0, 4.55], [0.3, 4.85, 4.85]],
                [[0.95, 1.4, 0.8], [1.85, 2.5, 4.8], [4.35, 4.25, 3.05]],
            ]
        ),
    )

##############################################################################
### TESTS/SUBMISSION CODE FOR backward passes


def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).data.sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).data.sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.data
            for x in out.op.gradient_as_tuple(dlspark.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.data for a in args]
    for i in range(len(numerical_grads)):
        print("numerical_grads", numerical_grads[i].shape)
        print("computed_grads", computed_grads[i].shape)
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_power_scalar_backward():
    gradient_check(
        dlspark.power_scalar, dlspark.Tensor(np.random.randn(5, 4)), scalar=np.random.randint(1)
    )


def test_divide_backward():
    gradient_check(
        dlspark.divide,
        dlspark.Tensor(np.random.randn(5, 4)),
        dlspark.Tensor(5 + np.random.randn(5, 4)),
    )


def test_divide_scalar_backward():
    gradient_check(
        dlspark.divide_scalar, dlspark.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1)
    )


def test_matmul_simple_backward():
    gradient_check(
        dlspark.matmul, dlspark.Tensor(np.random.randn(5, 4)), dlspark.Tensor(np.random.randn(4, 5))
    )


def test_matmul_batched_backward():
    gradient_check(
        dlspark.matmul,
        dlspark.Tensor(np.random.randn(6, 6, 5, 4)),
        dlspark.Tensor(np.random.randn(6, 6, 4, 3)),
    )
    gradient_check(
        dlspark.matmul,
        dlspark.Tensor(np.random.randn(6, 6, 5, 4)),
        dlspark.Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        dlspark.matmul,
        dlspark.Tensor(np.random.randn(5, 4)),
        dlspark.Tensor(np.random.randn(6, 6, 4, 3)),
    )


def test_reshape_backward():
    gradient_check(dlspark.reshape, dlspark.Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(dlspark.negate, dlspark.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(dlspark.transpose, dlspark.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(dlspark.transpose, dlspark.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))


def test_broadcast_to_backward():
    gradient_check(dlspark.broadcast_to, dlspark.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(dlspark.broadcast_to, dlspark.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(
        dlspark.broadcast_to,
        dlspark.Tensor(
            np.random.randn(
                1,
            )
        ),
        shape=(3, 3, 3),
    )
    gradient_check(dlspark.broadcast_to, dlspark.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(
        dlspark.broadcast_to, dlspark.Tensor(np.random.randn(5, 4, 1)), shape=(5, 4, 3)
    )


def test_summation_backward():
    gradient_check(dlspark.summation, dlspark.Tensor(np.random.randn(5, 4)), axes=(1,))
    gradient_check(dlspark.summation, dlspark.Tensor(np.random.randn(5, 4)), axes=(0,))
    gradient_check(dlspark.summation, dlspark.Tensor(np.random.randn(5, 4)), axes=(0, 1))
    gradient_check(dlspark.summation, dlspark.Tensor(np.random.randn(5, 4, 1)), axes=(0, 1))

##############################################################################
### TESTS/SUBMISSION CODE FOR find_topo_sort


def test_topo_sort():
    # Test case 1
    a1, b1 = dlspark.Tensor(np.asarray([[0.88282157]])), dlspark.Tensor(
        np.asarray([[0.90170084]])
    )
    c1 = 3 * a1 * a1 + 4 * b1 * a1 - a1

    soln = np.array(
        [
            np.array([[0.88282157]]),
            np.array([[2.64846471]]),
            np.array([[2.33812177]]),
            np.array([[0.90170084]]),
            np.array([[3.60680336]]),
            np.array([[3.1841638]]),
            np.array([[5.52228558]]),
            np.array([[-0.88282157]]),
            np.array([[4.63946401]]),
        ]
    )

    topo_order = np.array([x.data for x in dlspark.auto_grad.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = dlspark.Tensor(np.asarray([[0.20914675], [0.65264178]])), dlspark.Tensor(
        np.asarray([[0.65394286, 0.08218317]])
    )
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [
        np.array([[0.65394286, 0.08218317]]),
        np.array([[0.20914675], [0.65264178]]),
        np.array([[0.19040619]]),
        np.array([[1.53101102, 0.19240724]]),
        np.array([[0.44577898]]),
        np.array([[0.63618518]]),
        np.array([[1.90855553]]),
        np.array([[3.40855553]]),
    ]

    topo_order = [x.data for x in dlspark.auto_grad.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = dlspark.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = dlspark.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a @ b + b - a) @ a

    topo_order = np.array([x.data for x in dlspark.auto_grad.find_topo_sort([e])])

    soln = np.array(
        [
            np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
            np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
            np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
            np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
            np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
            np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
            np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]]),
        ]
    )

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

##############################################################################
### TESTS/SUBMISSION CODE FOR compute_gradient_of_variables


def test_compute_gradient():
    gradient_check(
        lambda A, B, C: dlspark.summation((A @ B + C) * (A @ B), axes=None),
        dlspark.Tensor(np.random.randn(10, 9)),
        dlspark.Tensor(np.random.randn(9, 8)),
        dlspark.Tensor(np.random.randn(10, 8)),
        backward=True,
    )
    gradient_check(
        lambda A, B: dlspark.summation(dlspark.broadcast_to(A, shape=(10, 9)) * B, axes=None),
        dlspark.Tensor(np.random.randn(10, 1)),
        dlspark.Tensor(np.random.randn(10, 9)),
        backward=True,
    )
    gradient_check(
        lambda A, B, C: dlspark.summation(
            dlspark.reshape(A, shape=(10, 10)) @ B / 5 + C, axes=None
        ),
        dlspark.Tensor(np.random.randn(100)),
        dlspark.Tensor(np.random.randn(10, 5)),
        dlspark.Tensor(np.random.randn(10, 5)),
        backward=True,
    )

    # check gradient of gradient
    x2 = dlspark.Tensor([6])
    x3 = dlspark.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.data
    x3_val = x3.data
    assert y.data == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.data == 2 * x2_val + x3_val
    assert grad_x3.data == x2_val
    assert grad_x2_x2.data == 2
    assert grad_x2_x3.data == 1


##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss


# def test_softmax_loss_spark():
#     # test forward pass for log
#     np.testing.assert_allclose(
#         dlspark.log(dlspark.Tensor([[4.0], [4.55]])).data,
#         np.array([[1.38629436112], [1.515127232963]]),
#     )

#     # test backward pass for log
#     gradient_check(dlspark.log, dlspark.Tensor(1 + np.random.rand(5, 4)))

#     X, y = parse_mnist(
#         "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
#     )
#     np.random.seed(0)
#     Z = dlspark.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
#     y_one_hot = np.zeros((y.shape[0], 10))
#     y_one_hot[np.arange(y.size), y] = 1
#     y = dlspark.Tensor(y_one_hot)
#     np.testing.assert_allclose(
#         softmax_loss(Z, y).data, 2.3025850, rtol=1e-6, atol=1e-6
#     )
#     Z = dlspark.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
#     np.testing.assert_allclose(
#         softmax_loss(Z, y).data, 2.7291998, rtol=1e-6, atol=1e-6
#     )

#     # test softmax loss backward
#     Zsmall = dlspark.Tensor(np.random.randn(16, 10).astype(np.float32))
#     ysmall = dlspark.Tensor(y_one_hot[:16])
#     gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)

# ##############################################################################
# ### TESTS/SUBMISSION CODE FOR nn_epoch


# def test_nn_epoch_spark():
#     # test forward/backward pass for relu
#     np.testing.assert_allclose(
#         dlspark.relu(
#             dlspark.Tensor(
#                 [
#                     [-46.9, -48.8, -45.45, -49.0],
#                     [-49.75, -48.75, -45.8, -49.25],
#                     [-45.65, -45.25, -49.3, -47.65],
#                 ]
#             )
#         ).data,
#         np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
#     )
#     gradient_check(dlspark.relu, dlspark.Tensor(np.random.randn(5, 4)))

#     # test nn gradients
#     np.random.seed(0)
#     X = np.random.randn(50, 5).astype(np.float32)
#     y = np.random.randint(3, size=(50,)).astype(np.uint8)
#     W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
#     W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
#     W1_0, W2_0 = W1.copy(), W2.copy()
#     W1 = dlspark.Tensor(W1)
#     W2 = dlspark.Tensor(W2)
#     X_ = dlspark.Tensor(X)
#     y_one_hot = np.zeros((y.shape[0], 3))
#     y_one_hot[np.arange(y.size), y] = 1
#     y_ = dlspark.Tensor(y_one_hot)
#     dW1 = nd.Gradient(
#         lambda W1_: softmax_loss(
#             dlspark.relu(X_ @ dlspark.Tensor(W1_).reshape((5, 10))) @ W2, y_
#         ).data
#     )(W1.data)
#     dW2 = nd.Gradient(
#         lambda W2_: softmax_loss(
#             dlspark.relu(X_ @ W1) @ dlspark.Tensor(W2_).reshape((10, 3)), y_
#         ).data
#     )(W2.data)
#     W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
#     np.testing.assert_allclose(
#         dW1.reshape(5, 10), W1_0 - W1.data, rtol=1e-4, atol=1e-4
#     )
#     np.testing.assert_allclose(
#         dW2.reshape(10, 3), W2_0 - W2.data, rtol=1e-4, atol=1e-4
#     )

#     # test full epoch
#     X, y = parse_mnist(
#         "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
#     )
#     np.random.seed(0)
#     W1 = dlspark.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
#     W2 = dlspark.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
#     W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
#     np.testing.assert_allclose(
#         np.linalg.norm(W1.data), 28.437788, rtol=1e-5, atol=1e-5
#     )
#     np.testing.assert_allclose(
#         np.linalg.norm(W2.data), 10.455095, rtol=1e-5, atol=1e-5
#     )
#     np.testing.assert_allclose(
#         loss_err(dlspark.relu(dlspark.Tensor(X) @ W1) @ W2, y),
#         (0.19770025, 0.06006667),
#         rtol=1e-4,
#         atol=1e-4,
#     )
