import torch
from torch import nn
import numpy as np
from torch.autograd import Variable 

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
    learning_rate_tensor = helper.make_tensor(learning_rate_name, onnx.TensorProto.FLOAT, dims=[], vals=[1.0])
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

########################################################################
### Test 1. Export a Pytorch linear model with its training information.
########################################################################
class LinearModel(nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x):
        return self.linear(x)



# Construct a Pytorch linear model and export it to ONNX.
N, D_in, D_out = 2, 1, 1
x = torch.Tensor([[1.0],[2.0]])
torch_model = LinearModel(D_in, D_out)
onnx_model_path = 'linear_model.onnx'

# ======= add training test in pytorch =========
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(torch_model.parameters(), lr = 0.01) 
x_data = Variable(torch.Tensor([[1.0], [2.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0]]))
for epoch in range(50): 
  # Forward pass: Compute predicted y by passing 
  # x to the model 
  pred_y = torch_model(x_data) 

  # Compute and print loss 
  loss = criterion(pred_y, y_data) 

  # Zero gradients, perform a backward pass, 
  # and update the weights. 
  optimizer.zero_grad() 
  loss.backward() 
  optimizer.step()
  print('epoch {}, loss {}'.format(epoch, loss.data)) 

new_data = Variable(torch.Tensor([[6.0], [8.0]]))
print('Predict after trainin for 6 and 8 is, ', torch_model(new_data))

# ========= end training in pytorch ==========

torch.onnx.export(torch_model, x, onnx_model_path, verbose=True)

# Load Pytorch ONNX model back in ONNX format and then append training information.
import onnx
from onnx import helper
onnx_model = onnx.load(onnx_model_path)

# Create label information manually.
label_tensor_name = 'label'

label_tensor_info = helper.make_tensor_value_info(label_tensor_name, onnx.TensorProto.FLOAT, shape=[N, 1])

# Create training information for the Pytorch model.
training_info = make_traning_components(onnx_model, label_tensor_info, onnx_model.graph.output[0], loss_type='MSE')
onnx_model.training_info.CopyFrom(training_info)

# Save ONNX model with training information.
onnx.save(onnx_model, 'training_' + onnx_model_path)
with open('training_linear_model.txt', 'wt') as f:
    f.write(str(onnx_model))

