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
    all_conv_nodes = []
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
            all_conv_nodes.append(model.graph.node[i])
            all_conv_node_indices.append(i)
    all_conv_nodes = np.array(all_conv_nodes)
    all_conv_node_indices = np.array(all_conv_node_indices)
    num_conv_nodes = all_conv_nodes.shape[0]
    weight_inits = model.graph.initializer
    for w in weight_inits:
        param_name_map[w.name] = w
    all_conv_node_indices_shuffled = all_conv_node_indices.copy()
    np.random.shuffle(all_conv_node_indices_shuffled)
    selected_conv_node_count = int(((100 - x) / 100.) * all_conv_node_indices_shuffled.shape[0])
    selected_conv_node_indices = np.sort(all_conv_node_indices_shuffled[:selected_conv_node_count])
    rejected_conv_node_indices = np.sort(all_conv_node_indices_shuffled[selected_conv_node_count:])
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
    print(onnx.helper.printable_graph(model.graph))

    pruned_model = prune(model, 20)
    k = 0
    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #