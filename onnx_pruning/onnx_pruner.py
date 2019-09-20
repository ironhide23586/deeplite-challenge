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

from collections import namedtuple

import numpy as np
import cv2

import torch
import torchvision

import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

import onnx
from onnx import shape_inference, numpy_helper, helper


def extract_linked_oneop_node_names(node_name_map, node_name):
    linked_node_names = []
    for k in node_name_map.keys():
        # Expand this exclusion list with 1-to-1 ops found at https://github.com/onnx/onnx/blob/master/docs/Operators.md
        if node_name in node_name_map[k].input and node_name_map[k].op_type in ['Relu', 'Dropout', 'MaxPool']:
            linked_node_names.append(k)
            linked_node_names += extract_linked_oneop_node_names(node_name_map, k)
    return linked_node_names


def extract_next_ingraph(node_name_map, node_name, rejected_node_names):
    node = node_name_map[node_name]
    if node_name not in rejected_node_names:
        return node
    for inp_name in node.input:
        if inp_name in node_name_map:
            return extract_next_ingraph(node_name_map, inp_name, rejected_node_names)
    return -1


def prune(model, x):
    """
    :param model: onnx model
    :param x: pruning ratio (0 to 100)
    :return: pruned model
    """
    shape_inferred_model = shape_inference.infer_shapes(model)
    shape_map = {}
    for i in range(len(shape_inferred_model.graph.value_info)):
        dims = [d.dim_value for d in shape_inferred_model.graph.value_info[i].type.tensor_type.shape.dim]
        shape_map[shape_inferred_model.graph.value_info[i].name] = dims
    num_nodes = len(model.graph.node)
    all_conv_node_names = []
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
            all_conv_node_names.append(model.graph.node[i].name)
    all_conv_node_names = np.array(all_conv_node_names)

    for param in model.graph.initializer:
        param_name_map[param.name] = param

    for inp in model.graph.input:
        input_name_map[inp.name] = inp

    all_conv_node_names_shuffled = list(all_conv_node_names.copy())
    np.random.shuffle(all_conv_node_names_shuffled)
    selected_conv_node_count = int(((100 - x) / 100.) * len(all_conv_node_names_shuffled))
    rejected_node_names = all_conv_node_names_shuffled[selected_conv_node_count:]
    for name in rejected_node_names:  # finding full sets of nodes qualified for removal
        linked_oneop_node_names = extract_linked_oneop_node_names(node_name_map, name)
        rejected_node_names += linked_oneop_node_names
    new_nn_nodes = []
    new_nn_input_names = []
    for name in node_name_map.keys():  # identifying required input nodes in the new neural net
        if name in rejected_node_names:
            continue
        else:
            new_nn_nodes.append(node_name_map[name])
            input_names = node_name_map[name].input
            for input_name in input_names:
                if input_name in input_name_map:
                    new_nn_input_names.append(input_name)
    new_nn_input_names = list(set(new_nn_input_names))
    new_nn_inputs = [input_name_map[name] for name in new_nn_input_names]
    new_nn_params = [param_name_map[name] for name in new_nn_input_names if name != model.graph.input[0].name]

    for i in range(len(new_nn_nodes)):  # rewiring the neural net to fill gaps created by missing layers
        node = new_nn_nodes[i]
        if len(node.input) == 0:
            continue
        for j in range(len(node.input)):
            inp_name = node.input[j]
            if inp_name in rejected_node_names:
                ingraph_input_node = extract_next_ingraph(node_name_map, inp_name, rejected_node_names)
                node.input[j] = ingraph_input_node.name
                input_shape = shape_map[ingraph_input_node.name]
                output_shape = shape_map[new_nn_nodes[i].name]
                in_channels = input_shape[1]
                out_channels = output_shape[1]
                for weight_name in node.input:
                    if weight_name in param_name_map and len(param_name_map[weight_name].dims) == 4:
                        conv_param_name = weight_name
                conv_param = param_name_map[conv_param_name]
                conv_input = input_name_map[conv_param_name]
                conv_filter_dims = conv_param.dims  # in_channels, out_channels, kernel_height, kernel_width
                if conv_filter_dims[0] != in_channels:
                    filter_values = numpy_helper.to_array(conv_param)
                    k = 0
        k = 0
        # input_node_name = new_nn_nodes[i].input[0]
        # while input_node_name in rejected_node_names:
        #     if len(node_name_map[input_node_name].input) > 0:
        #         input_node_name = node_name_map[input_node_name].input[0]
        #     else:
        #         input_node_name = ''
        # if len(input_node_name) > 0:
        #     new_nn_nodes[i].input[0] = input_node_name
        # else:
        #     new_nn_nodes[i].input = new_nn_nodes[i].input[1:]

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
    # print('PyTorch classified label -', out_label)
    # input_names = ['image_input']
    # output_names = ['logit_outs']
    # torch.onnx.export(model, dummy_input, "vgg19.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names, export_params=True)
    # ----------------- CREATION OF ONNX MODEL FROM PRETRAINED PYTORCH MODEL ----------------- #

    # -------------------------------- OPERATING ON ONNX MODEL -------------------------------- #
    model = onnx.load('vgg19.onnx')
    onnx.checker.check_model(model)
    print('Original model -')
    print(helper.printable_graph(model.graph))

    pruned_model = prune(model, 20)
    print('Pruned model -')
    print(helper.printable_graph(pruned_model.graph))
    onnx.save_model(pruned_model, 'vgg19_pruned.onnx')
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
    sym, arg, aux = onnx_mxnet.import_model('vgg19_pruned.onnx')
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    param_names_args = [graph_input for graph_input in arg if graph_input not in data_names]
    param_names_aux = [graph_input for graph_input in aux if graph_input not in data_names]
    param_shapes_args = [arg[n].shape for n in param_names_args]
    param_shapes_aux = [arg[n].shape for n in param_names_aux]

    param_names_all = param_names_args + param_names_aux
    param_shapes_all = param_shapes_args + param_shapes_aux

    all_data_names = [data_names[0]] + param_names_all
    all_data_shapes = [im.shape] + param_shapes_all

    all_data_shape_list = list(zip(all_data_names, all_data_shapes))

    # mod = mx.mod.Module(symbol=sym, data_names=all_data_names, context=mx.gpu(), label_names=None)
    mod = mx.mod.Module(symbol=sym, data_names=[data_names[0]], context=mx.gpu(), label_names=None)

    # mod.bind(for_training=False, data_shapes=all_data_shape_list, label_shapes=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], im.shape)], label_shapes=None)

    mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(im)]))
    out_logits = mod.get_outputs()[0].asnumpy()[0].argmax()
    out_label = logitmap[out_logits]
    print('MXnet classified label from convertted ONNX model -', out_label)

    k = 0
    # ----------------- LOADING PRUNED ONNX MODEL AND VALIDATION IN MXNET ----------------- #