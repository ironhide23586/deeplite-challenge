"""
    Deeplite coding challenge: onnx_pruner.
    The main goal of this coding challenge is to implement a method to prune conv layers of a given onnx model.

    Details:
    Take an onnx model and randomly remove x percent (x is a given number between 0 to 100) of conv layers in such
    a way that the new onnx model is still valid and you can train/test it.

    ** First select the random conv layers for pruning then remove them one by one (sequentially)
    ** You may need to adjust the input/output of remaining layers after each layer pruning
    ** you can test your code on vgg19
    ** Can you extend your work to support models with residual connections such as resnet family?
    ** We recommend using mxnet as the AI framework for this coding challenge due to its native support of onnx
       https://mxnet.incubator.apache.org/versions/master/api/python/contrib/onnx.html
"""

import numpy as np

import torch
import torchvision

import onnx
from onnx import shape_inference, numpy_helper, helper


def prune(model, x):
    """
    :param model: onnx model
    :param x: pruning ratio (0 to 100)
    :return: pruned model
    """
    num_nodes = len(model.graph.node)
    all_conv_node_indices = []
    plural_nodes = []
    node_name_map = {}
    param_name_map = {}
    for i in range(num_nodes):
        model.graph.node[i].name = model.graph.node[i].output[0]
        node_name_map[model.graph.node[i].name] = model.graph.node[i]
        if len(model.graph.node[i].output) > 1:
            plural_nodes.append(model.graph.node[i])
        if model.graph.node[i].op_type == 'Conv':
            all_conv_node_indices.append(i)
    all_conv_node_indices = np.array(all_conv_node_indices)
    weight_inits = model.graph.initializer
    for w in weight_inits:
        param_name_map[w.name] = w
    all_conv_node_indices_shuffled = all_conv_node_indices.copy()
    np.random.shuffle(all_conv_node_indices_shuffled)
    selected_conv_node_count = int(((100 - x) / 100.) * all_conv_node_indices_shuffled.shape[0])
    rejected_node_indices = all_conv_node_indices_shuffled[selected_conv_node_count:]
    rejected_node_names = []
    for idx in rejected_node_indices:  # finding full sets of nodes qualified for removal
        rejected_node_names.append(model.graph.node[idx].name)
        idx_local = idx + 1
        while model.graph.node[idx_local].op_type in ['Relu', 'Dropout', 'MaxPool']:  # Expand this list with 1-to-1 ops
            rejected_node_indices = np.hstack([rejected_node_indices, [idx_local]])
            rejected_node_names.append(model.graph.node[idx_local].name)
            idx_local += 1
    new_nn_nodes = []
    new_nn_param_names = []
    for i in range(num_nodes):  # identifying required param nodes in the new neural net
        if i in rejected_node_indices:
            continue
        else:
            new_nn_nodes.append(model.graph.node[i])
            param_names = model.graph.node[i].input[1:]
            for name in param_names:
                if name in param_name_map:
                    new_nn_param_names.append(name)
    new_nn_param_names = list(set(new_nn_param_names))
    new_nn_params = [param_name_map[name] for name in new_nn_param_names]
    for i in range(len(new_nn_nodes)):  # rewiring the neural net to fill gaps created by missing layers
        if len(new_nn_nodes[i].input) == 0:
            continue
        input_node_name = new_nn_nodes[i].input[0]
        while input_node_name in rejected_node_names:
            if len(node_name_map[input_node_name].input) > 0:
                input_node_name = node_name_map[input_node_name].input[0]
            else:
                input_node_name = ''
        if len(input_node_name) > 0:
            new_nn_nodes[i].input[0] = input_node_name
        else:
            new_nn_nodes[i].input = new_nn_nodes[i].input[1:]
    k = 0


if __name__ == '__main__':

    # ----------------- CREATION OF ONNX MODEL FROM PRETRAINED PYTORCH MODEL ----------------- #
    # model = torchvision.models.vgg19(pretrained=True).cuda()
    # dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    # dummy_output = model.forward(dummy_input)
    # input_names = ['image_input']
    # output_names = ['logit_outs']
    # torch.onnx.export(model, dummy_input, "vgg19.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names, export_params=True)
                      # dynamic_axis={'image_input': {0: 'batch_size'}, 'logit_outs': {0: 'batch_size'}})
    # ----------------- CREATION OF ONNX MODEL FROM PRETRAINED PYTORCH MODEL ----------------- #

    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #
    model = onnx.load('vgg19.onnx')
    onnx.checker.check_model(model)
    print(helper.printable_graph(model.graph))

    pruned_model = prune(model, 20)
    k = 0
    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #