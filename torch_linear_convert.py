import torch
from torch import nn
import numpy as np
from torch.autograd import Variable 

def make_function(name, inputs, outputs, nodes, attributes=None):
    algorithm = onnx.FunctionProto()
    algorithm.name = name
    algorithm.input.extend(inputs)
    algorithm.output.extend(outputs)
    algorithm.node.extend(nodes)
    if attributes is not None:
        algorithm.attribute.extend(attributes)
    return algorithm

def make_training_info(initializers, inputs, outputs,
        algorithm, update_bindings):
    training_info = onnx.TrainingInfoProto()
    training_info.initializer.extend(initializers)
    training_info.input.extend(inputs)
    training_info.output.extend(outputs)
    training_info.algorithm.CopyFrom(algorithm)
    for k, v in update_bindings.items():
        bind = training_info.update_binding.add()
        bind.key = k
        bind.value = v
    return training_info

def make_traning_components(onnx_model, label_tensor_info, score_tensor_info, loss_type, inference_function_name='MyInferenceFunction'):
    additional_inputs = [label_tensor_info]
    additional_outputs = []
    additional_initializers = []

    update_bindings = {}

    # Call inference graph (type: FunctionProto) in the training algorithm.
    inference_initializer_names = list(info.name for info in onnx_model.graph.initializer)
    inference_total_inputs = list(info.name for info in onnx_model.graph.input if info.name not in inference_initializer_names) + inference_initializer_names
    inference_node = helper.make_node(op_type=inference_function_name,
                                      inputs=inference_total_inputs,
                                      outputs=[info.name for info in onnx_model.graph.output])

    # Compute loss using outputs produced by the inference call.
    loss_tensor_name = 'loss_value'
    if score_tensor_info.name not in [info.name for info in onnx_model.graph.output]:
        raise Exception('Score cannot be found among inference outputs.')
    loss_node = helper.make_node(op_type=loss_type, inputs=[label_tensor_info.name, score_tensor_info.name],
                                 outputs=[loss_tensor_name])
    loss_tensor_info = helper.make_tensor_value_info(loss_tensor_name, onnx.TensorProto.FLOAT, shape=[])
    additional_outputs.append(loss_tensor_info)


    gradient_node = helper.make_node(op_type='Gradient',
                                     inputs=[info.name for info in onnx_model.graph.input],
                                     outputs=['grad_' + info.name for info in onnx_model.graph.initializer],
                                     xs=[info.name for info in onnx_model.graph.initializer],
                                     y=loss_tensor_name)

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

    optimized_tensors = []
    gradient_tensors = []
    state_tensors = []
    new_optimized_tensors = []
    new_state_tensors = []

    for info in onnx_model.graph.initializer:
        gradient_tensor_name = 'grad_' + info.name

        state_tensor_name = 'squared_grad_sum' + info.name
        state_initializer = helper.make_tensor(state_tensor_name, onnx.TensorProto.FLOAT, dims=info.dims,
                                               vals=np.ones(info.dims).flatten())
        additional_initializers.append(state_initializer)

        update_tensor_name = 'new_' + info.name
        update_tensor_info = helper.make_tensor_value_info(update_tensor_name, info.data_type, shape=info.dims)
        additional_outputs.append(update_tensor_info)
        update_bindings[update_tensor_name] = info.name

        new_state_tensor_name = 'new_squared_grad_sum_' + info.name
        new_state_info = helper.make_tensor_value_info(new_state_tensor_name, info.data_type, shape=info.dims)
        additional_outputs.append(new_state_info)
        update_bindings[new_state_tensor_name] = state_tensor_name

        optimized_tensors.append(info.name)
        gradient_tensors.append(gradient_tensor_name)
        state_tensors.append(state_tensor_name)
        new_optimized_tensors.append(update_tensor_name)
        new_state_tensors.append(new_state_tensor_name)

    optimizer_node = helper.make_node(op_type='Adagrad',
                                      inputs=optimizer_inputs + optimized_tensors + gradient_tensors + state_tensors,
                                      outputs=new_optimized_tensors + new_state_tensors,
                                      name='Optimizer', **optimizer_attrs)

    total_nodes = [inference_node, loss_node, gradient_node, optimizer_node]
    total_initializers = list(onnx_model.graph.initializer) + additional_initializers
    inference_initializer_names = set(info.name for info in onnx_model.graph.initializer)
    total_inputs = list(info for info in onnx_model.graph.input if info.name not in inference_initializer_names) + additional_inputs
    total_outputs = additional_outputs

    algorithm = make_function('MyTrainingFunction', [info.name for info in total_inputs] + [info.name for info in total_initializers], new_optimized_tensors + new_state_tensors, total_nodes)

    return make_training_info(initializers=additional_initializers,
                              inputs=total_inputs,
                              outputs=total_outputs,
                              algorithm=algorithm,
                              update_bindings=update_bindings)

def move_inference_graph_to_function(onnx_model):
    inference_initializer_names = list(info.name for info in onnx_model.graph.initializer)
    input_names = [info.name for info in onnx_model.graph.input if info.name not in inference_initializer_names] + inference_initializer_names
    output_names = [info.name for info in onnx_model.graph.output]
    inference_function = make_function('MyInferenceFunction', input_names, output_names, list(onnx_model.graph.node))
    for i in range(len(onnx_model.graph.node)):
        onnx_model.graph.node.pop()
    for i in range(len(onnx_model.graph.value_info)):
        onnx_model.graph.value_info.pop()
    onnx_model.function.extend([inference_function])

    # Call inference graph (type: FunctionProto) in the training algorithm.
    inference_node = helper.make_node(op_type='MyInferenceFunction',
                                      inputs=input_names,
                                      outputs=[info.name for info in onnx_model.graph.output])
    onnx_model.graph.node.extend([inference_node])

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
N, D_in, D_out = 5, 1, 1
x = torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
torch_model = LinearModel(D_in, D_out)
onnx_model_path = 'linear_model.onnx'

# ======= add training test in pytorch =========
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(torch_model.parameters(), lr = 0.01) 
x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]]))
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0],[8.0],[10.0]]))
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

new_data = Variable(torch.Tensor([[11.0],[12.0],[13.0],[14.0],[15.0]]))
print('Predict after trainin for 11-15 is, ', torch_model(new_data))

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
training_info_buffer = onnx_model.training_info.add()
training_info_buffer.CopyFrom(training_info)
move_inference_graph_to_function(onnx_model)
#print(onnx_model)

# Save ONNX model with training information.
onnx.save(onnx_model, 'training_' + onnx_model_path)
with open('training_linear_model.txt', 'wt') as f:
    f.write(str(onnx_model))

