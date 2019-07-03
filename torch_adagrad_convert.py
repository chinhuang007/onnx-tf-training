import tensorflow as tf
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable 
import torch.random as rnd
from torch import optim

def make_training_info(
        additional_initializers, additional_inputs, additional_outputs,
        loss, gradient_bindings, optimizer, update_bindings):
    training_info = onnx.TrainingInfoProto()
    training_info.additional_initializer.extend(additional_initializers)
    training_info.additional_input.extend(additional_inputs)
    training_info.additional_output.extend(additional_outputs)
    training_info.loss.CopyFrom(loss)
    for k, v in gradient_bindings.items():
        bind = training_info.gradient_binding.add()
        bind.key = k
        bind.value = v
    training_info.optimizer.CopyFrom(optimizer)
    for k, v in update_bindings.items():
        bind = training_info.update_binding.add()
        bind.key = k
        bind.value = v
    return training_info

def make_traning_components(onnx_model, label_tensor_info, score_tensor_info, loss_type):
    additional_inputs = [label_tensor_info]
    additional_outputs = []
    additional_initializers = []

    gradient_bindings = {}
    update_bindings = {}

    # Adagrad iteartion count.
    iteration_count_tensor_name = 'T'
    iteration_count_tensor = helper.make_tensor(iteration_count_tensor_name, onnx.TensorProto.INT64, dims=[], vals=[0])
    additional_initializers.append(iteration_count_tensor)

    # Adagrad learning rate.
    learning_rate_name = 'R'
    learning_rate_tensor = helper.make_tensor(learning_rate_name, onnx.TensorProto.FLOAT, dims=[], vals=[0.2])
    additional_initializers.append(learning_rate_tensor)

    optimizer_inputs = [learning_rate_name, iteration_count_tensor_name]
    optimizer_attrs = {'norm_coefficient': 0.0, 'epsilon': 1e-7, 'decay_factor': 0.0}
    optimizer_outputs = []

    for info in onnx_model.graph.initializer:
        gradient_tensor_name = 'grad_' + info.name
        gradient_bindings[info.name] = gradient_tensor_name

        state_tensor_name = 'sq_grad_sum' + info.name
        state_initializer = helper.make_tensor(state_tensor_name, onnx.TensorProto.FLOAT, dims=info.dims,
                                               vals=np.ones(info.dims).flatten())
        additional_initializers.append(state_initializer)

        update_tensor_name = 'new_' + info.name
        update_tensor_info = helper.make_tensor_value_info(update_tensor_name, info.data_type, shape=info.dims)
        additional_outputs.append(update_tensor_info)
        update_bindings[info.name] = update_tensor_name

        new_state_tensor_name = 'new_sq_grad_sum_' + info.name
        new_state_info = helper.make_tensor_value_info(new_state_tensor_name, info.data_type, shape=info.dims)
        additional_outputs.append(new_state_info)
        update_bindings[state_tensor_name] = new_state_tensor_name

        optimizer_inputs.append(info.name)
        optimizer_inputs.append(gradient_tensor_name)
        optimizer_inputs.append(state_tensor_name)
        optimizer_outputs.append(update_tensor_name)
        optimizer_outputs.append(new_state_tensor_name)

    optimizer_node = helper.make_node(op_type='Adagrad',
                                      inputs=optimizer_inputs,
                                      outputs=optimizer_outputs,
                                      name='Optimizer', **optimizer_attrs)

    loss_tensor_name = 'loss_value'
    loss_node = helper.make_node(op_type=loss_type, inputs=[label_tensor_info.name, score_tensor_info.name],
                                 outputs=[loss_tensor_name])
    loss_tensor_info = helper.make_tensor_value_info(loss_tensor_name, onnx.TensorProto.FLOAT, shape=[])
    additional_outputs.append(loss_tensor_info)

    return make_training_info(additional_initializers=additional_initializers,
                              additional_inputs=additional_inputs,
                              additional_outputs=additional_outputs,
                              loss=loss_node,
                              gradient_bindings=gradient_bindings,
                              optimizer=optimizer_node,
                              update_bindings=update_bindings)

# Construct a Pytorch linear model and export it to ONNX.
onnx_model_path = 'linear_model.onnx'

lr=0.2
lr_decay=0.0
weight_decay=0.0
n = 1 
l = 5
#X_ = torch.randn(l, n)
#Y_ = torch.randn(l, 1)
# use a randomly generated dataset as constants to verify results
X_ = torch.tensor([[-1.4027],
        [-0.7377],
        [-0.5206],
        [ 0.2311],
        [-0.5809]])
Y_ = torch.tensor([[ 1.3886],
        [ 0.7485],
        [ 1.9602],
        [-1.4880],
        [ 1.0679]])

onnx_model_path = 'linear_model.onnx'

def show_pytorch():
    X = X_.clone()
    Y = Y_.clone()

    rnd.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(n, 1, bias=False)
    )

    loss_fn = nn.MSELoss(reduction='sum')

    solver = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)

    new_data = torch.tensor([[10],[20],[30],[40],[50]], dtype=torch.float32)
    print('Predict before training for [10, 20, 30, 40, 50] is, ', model(new_data))
    torch.onnx.export(model, X, onnx_model_path, verbose=True)

    for t in range(50):
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        model.zero_grad()
        loss.backward()
        solver.step()
    print('Predict after training for [10, 20, 30, 40, 50] is, ', model(new_data))

def show_tensorflow():
    rnd.manual_seed(0)
    layer = nn.Linear(n, 1, bias=False)

    new_data = [[10],[20],[30],[40],[50]]
    X = tf.placeholder('float', shape=[l, n])
    W = tf.Variable(torch.Tensor.numpy(layer.weight.detach()))
    Y = tf.placeholder('float', shape=[l, 1])
    Y_pred = tf.matmul(X, W, transpose_b=True)
    loss = tf.reduce_sum(tf.square(Y - Y_pred))
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    minimizer = optimizer.minimize(loss)

    sess = tf.Session()

    init = tf.global_variables_initializer()

    sess.run(init)
    print('Predict before training for [10, 20, 30, 40, 50] is, ', sess.run(Y_pred, {X: new_data}))

    for t in range(50):
        result = sess.run([minimizer], {X: X_, Y: Y_})

    print('Predict after training for [10, 20, 30, 40, 50] is, ', sess.run(Y_pred, {X: new_data}))
show_pytorch()
print('---------')

show_tensorflow()
print('---------')

# Load Pytorch ONNX model back in ONNX format and then append training information.
import onnx
from onnx import helper
onnx_model = onnx.load(onnx_model_path)

# Create label information manually.
label_tensor_name = 'label'
N = 5
label_tensor_info = helper.make_tensor_value_info(label_tensor_name, onnx.TensorProto.FLOAT, shape=[N, 1])

# Create training information for the Pytorch model.
training_info = make_traning_components(onnx_model, label_tensor_info, onnx_model.graph.output[0], loss_type='MSE')
onnx_model.training_info.CopyFrom(training_info)

# Save ONNX model with training information.
onnx.save(onnx_model, 'training_' + onnx_model_path)
with open('training_linear_model.txt', 'wt') as f:
    f.write(str(onnx_model))

