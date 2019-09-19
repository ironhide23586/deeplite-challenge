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
import cv2

import torch
import torchvision

import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

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
    input_name_map = {}
    for i in range(num_nodes):
        model.graph.node[i].name = model.graph.node[i].output[0]
        node_name_map[model.graph.node[i].name] = model.graph.node[i]
        if len(model.graph.node[i].output) > 1:
            plural_nodes.append(model.graph.node[i])
        if model.graph.node[i].op_type == 'Conv':
            all_conv_node_indices.append(i)
    all_conv_node_indices = np.array(all_conv_node_indices)

    for param in model.graph.initializer:
        param_name_map[param.name] = param

    for inp in model.graph.input:
        input_name_map[inp.name] = inp

    all_conv_node_indices_shuffled = all_conv_node_indices.copy()
    np.random.shuffle(all_conv_node_indices_shuffled)
    selected_conv_node_count = int(((100 - x) / 100.) * all_conv_node_indices_shuffled.shape[0])
    rejected_node_indices = all_conv_node_indices_shuffled[selected_conv_node_count:]
    rejected_node_names = []
    for idx in rejected_node_indices:  # finding full sets of nodes qualified for removal
        rejected_node_names.append(model.graph.node[idx].name)
        idx_local = idx + 1

        # Expand this exclusion list with 1-to-1 ops found at https://github.com/onnx/onnx/blob/master/docs/Operators.md
        while model.graph.node[idx_local].op_type in ['Relu', 'Dropout', 'MaxPool']:
            rejected_node_indices = np.hstack([rejected_node_indices, [idx_local]])
            rejected_node_names.append(model.graph.node[idx_local].name)
            idx_local += 1
    new_nn_nodes = []
    new_nn_input_names = []
    for i in range(num_nodes):  # identifying required input nodes in the new neural net
        if i in rejected_node_indices:
            continue
        else:
            new_nn_nodes.append(model.graph.node[i])
            input_names = model.graph.node[i].input[1:]
            for name in input_names:
                if name in input_name_map:
                    new_nn_input_names.append(name)
    new_nn_input_names = list(set(new_nn_input_names))
    new_nn_inputs = [input_name_map[name] for name in new_nn_input_names]
    new_nn_params = [param_name_map[name] for name in new_nn_input_names if name != model.graph.input[0].name]
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
    new_nn_graph = helper.make_graph(
        new_nn_nodes,
        "ConvNet-trimmed",
        [model.graph.input[0]] + new_nn_inputs,
        model.graph.output,
        new_nn_params
    )
    new_nn_model = helper.make_model(new_nn_graph)
    onnx.checker.check_model(new_nn_model)
    # shape_inferred_model = shape_inference.infer_shapes(new_nn_model)
    return new_nn_model


def logit2class_mapper():
    with open('imagenet_classes.txt', 'r') as f:
        class_keys = f.readlines()
        class_keys = [cls.strip() for cls in class_keys]
    with open('imagenet_synsets.txt', 'r') as f:
        data = f.readlines()
        key2label_map = {d.strip().split(' ')[0]: ' '.join(d.strip().split(' ')[1:]) for d in data}
    logitmap = [key2label_map[k] for k in class_keys]
    return logitmap


if __name__ == '__main__':

    logitmap = logit2class_mapper()
    im = cv2.resize(cv2.imread('car.jpg'), (224, 224))[:, :, [2, 1, 0]]
    im = (np.rollaxis(im, 2, 0) / 255.).astype(np.float32)
    im = np.expand_dims(im, 0)

    # ----------------- CREATION OF ONNX MODEL FROM PRETRAINED PYTORCH MODEL ----------------- #
    # model = torchvision.models.vgg19(pretrained=True).cuda()
    # dummy_input = torch.from_numpy(im).cuda()
    # dummy_output = model.forward(dummy_input).cpu().detach().numpy()[0]
    # out_label = logitmap[dummy_output.argmax()]
    # print(out_label)
    # input_names = ['image_input']
    # output_names = ['logit_outs']
    # torch.onnx.export(model, dummy_input, "vgg19.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names, export_params=True)
    # ----------------- CREATION OF ONNX MODEL FROM PRETRAINED PYTORCH MODEL ----------------- #

    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #
    # model = onnx.load('vgg19.onnx')
    # onnx.checker.check_model(model)
    # print('Original model -')
    # print(helper.printable_graph(model.graph))
    #
    # pruned_model = prune(model, 20)
    # print('Pruned model -')
    # print(helper.printable_graph(pruned_model.graph))
    # onnx.save_model(pruned_model, 'vgg19_pruned.onnx')
    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #

    # ----------------- LOADING PRUNED ONNX MODEL AND VALIDATION IN MXNET ----------------- #
    # KeyError: 'concat0' -> https://github.com/apache/incubator-mxnet/issues/13949 (open GitHub issue)
    # modified mxnet source code at -
    # https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/onnx/onnx2mx/_op_translations.py#L462
    # to accommodate for missing shape of resulting tensor after reshaping.
    # Source code changed -
    # reshape_shape = list(proto_obj._params[inputs[1].name].asnumpy()) (previous)
    # reshape_shape = [1, 25088] (replaced with this, hacky solution for now)
    # Ideally this should be a flatten operator but PyTorch exports it as a reshape :(
    sym, arg, aux = onnx_mxnet.import_model('vgg19.onnx')
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.gpu(), label_names=logitmap)

    mod.bind(for_training=False, data_shapes=[(data_names[0], im.shape)], label_shapes=None)
    k = 0
    # ----------------- LOADING PRUNED ONNX MODEL AND VALIDATION IN MXNET ----------------- #