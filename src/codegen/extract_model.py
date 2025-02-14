import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import absl.logging
import warnings
from tensorflow import keras

absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getAlphaForActivation(layer, activation):
    if isinstance(activation, dict) and activation.get('class_name') == 'LeakyReLU':
        return activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01))
    elif activation == 'elu':
        return layer.get_config().get('alpha', 1.0)
    return 0.0


def extractModel(model, file_type):
    weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params = [], [], [], [], [], [], []

    if file_type in ['.h5', '.keras']:
        for layer in model.layers:
            layer_weights = layer.get_weights()

            # ===== NEW CODE: Convolution Layers Extraction =====
            # Supports: Conv2D, SeparableConv2D, ConvLSTM2D, Conv1D, Conv3D, Conv2DTranspose, DepthwiseConv2D, LocallyConnected2D
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D, keras.layers.ConvLSTM2D,
                                  keras.layers.Conv1D, keras.layers.Conv3D, keras.layers.Conv2DTranspose,
                                  keras.layers.DepthwiseConv2D)):
                config = layer.get_config()
                conv_param = {}
                conv_param["layer_type"] = layer.__class__.__name__
                # Try to get input_shape
                try:
                    conv_param["input_shape"] = layer.input_shape
                except AttributeError:
                    conv_param["input_shape"] = getattr(layer, "_batch_input_shape", None)
                if conv_param["input_shape"] is None and hasattr(layer, "input") and layer.input is not None:
                    if isinstance(layer.input, tuple):
                        conv_param["input_shape"] = list(layer.input[0].shape)
                    else:
                        try:
                            conv_param["input_shape"] = layer.input.shape.as_list()
                        except AttributeError:
                            conv_param["input_shape"] = list(layer.input.shape)
                # Try to get output_shape similarly
                try:
                    conv_param["output_shape"] = layer.output_shape
                except AttributeError:
                    conv_param["output_shape"] = getattr(layer, "_batch_output_shape", None)
                if conv_param["output_shape"] is None and hasattr(layer, "output") and layer.output is not None:
                    if isinstance(layer.output, tuple):
                        conv_param["output_shape"] = list(layer.output[0].shape)
                    else:
                        try:
                            conv_param["output_shape"] = layer.output.shape.as_list()
                        except AttributeError:
                            conv_param["output_shape"] = list(layer.output.shape)
                conv_param["strides"] = config.get("strides", (1, 1))
                conv_param["padding"] = config.get("padding", "valid")
                conv_param["dilation_rate"] = config.get("dilation_rate", (1, 1))
                # For SeparableConv2D, extract depthwise and pointwise kernels separately.
                if isinstance(layer, keras.layers.SeparableConv2D):
                    if len(layer_weights) > 0:
                        depthwise_kernel = layer_weights[0]
                        conv_param["depthwise_kernel"] = depthwise_kernel.flatten()
                        conv_param["depthwise_kernel_shape"] = depthwise_kernel.shape
                    else:
                        conv_param["depthwise_kernel"] = None
                        conv_param["depthwise_kernel_shape"] = None
                    if len(layer_weights) > 1:
                        pointwise_kernel = layer_weights[1]
                        conv_param["pointwise_kernel"] = pointwise_kernel.flatten()
                        conv_param["pointwise_kernel_shape"] = pointwise_kernel.shape
                    else:
                        conv_param["pointwise_kernel"] = None
                        conv_param["pointwise_kernel_shape"] = None
                    if len(layer_weights) > 2:
                        bias = layer_weights[2]
                        conv_param["bias"] = bias.flatten()
                        conv_param["bias_shape"] = bias.shape
                    else:
                        conv_param["bias"] = None
                        conv_param["bias_shape"] = None
                else:
                    # For standard convolutional layers
                    if len(layer_weights) > 0:
                        kernel = layer_weights[0]
                        conv_param["kernel"] = kernel.flatten()
                        conv_param["kernel_shape"] = kernel.shape
                    else:
                        conv_param["kernel"] = None
                        conv_param["kernel_shape"] = None
                    if len(layer_weights) > 1:
                        bias = layer_weights[1]
                        conv_param["bias"] = bias.flatten()
                        conv_param["bias_shape"] = bias.shape
                    else:
                        conv_param["bias"] = None
                        conv_param["bias_shape"] = None

                conv_param["activation"] = config.get("activation", "linear")
                # Append dummy values for the other lists and add conv parameters
                weights_list.append(None)
                biases_list.append(None)
                activation_functions.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                norm_layer_params.append(None)
                conv_layer_params.append(conv_param)
                continue  # Skip to next layer

            # ===== NEW CODE: Pooling Layers Extraction =====
            # Supports: MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling1D, AveragePooling1D
            elif isinstance(layer, (keras.layers.MaxPooling2D, keras.layers.AveragePooling2D,
                                    keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D,
                                    keras.layers.MaxPooling1D, keras.layers.AveragePooling1D)):
                config = layer.get_config()
                pool_param = {}
                pool_param["layer_type"] = layer.__class__.__name__
                try:
                    pool_param["input_shape"] = layer.input_shape
                except AttributeError:
                    pool_param["input_shape"] = getattr(layer, "_batch_input_shape", None)
                if pool_param["input_shape"] is None and hasattr(layer, "input") and layer.input is not None:
                    if isinstance(layer.input, tuple):
                        pool_param["input_shape"] = list(layer.input[0].shape)
                    else:
                        try:
                            pool_param["input_shape"] = layer.input.shape.as_list()
                        except AttributeError:
                            pool_param["input_shape"] = list(layer.input.shape)
                try:
                    pool_param["output_shape"] = layer.output_shape
                except AttributeError:
                    pool_param["output_shape"] = getattr(layer, "_batch_output_shape", None)
                if pool_param["output_shape"] is None and hasattr(layer, "output") and layer.output is not None:
                    if isinstance(layer.output, tuple):
                        pool_param["output_shape"] = list(layer.output[0].shape)
                    else:
                        try:
                            pool_param["output_shape"] = layer.output.shape.as_list()
                        except AttributeError:
                            pool_param["output_shape"] = list(layer.output.shape)
                pool_param["pool_size"] = config.get("pool_size", None)
                pool_param["strides"] = config.get("strides", None)
                pool_param["padding"] = config.get("padding", "valid")
                # Append dummy values for the other lists and add pool parameters
                weights_list.append(None)
                biases_list.append(None)
                activation_functions.append("pooling")  # marker for pooling layers
                alphas.append(0.0)
                dropout_rates.append(0.0)
                norm_layer_params.append(None)
                conv_layer_params.append(pool_param)
                continue

            # ===== Original Code (Dense, Activations, Norm, etc.) =====
            # --- OLD CODE (preserved) ---
            """
            conv_layer_params.append(None)
            if 'activation' in layer.name.lower() or isinstance(layer, keras.layers.Activation):
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
            elif 'flatten' in layer.name.lower() or isinstance(layer, keras.layers.Flatten):
                activation = 'flatten'
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
            elif 'batch_normalization' in layer.name.lower() or isinstance(layer, keras.layers.BatchNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    norm_layer_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append('batchNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
            elif 'layer_normalization' in layer.name.lower() or isinstance(layer, keras.layers.LayerNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    norm_layer_params.append((gamma, beta, None, None, epsilon))
                    activation_functions.append('layerNormalization')
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
            else:
                if len(layer_weights) == 2:
                    weights, biases = layer_weights
                else:
                    weights, biases = None, None
                weights_list.append(weights)
                biases_list.append(biases)
                norm_layer_params.append(None)
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
                activation_functions.append(activation if activation != 'linear' else 'linear')
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)
            """
            # --- NEW CODE: (For Dense and Other Layers) ---
            weights_list.append(layer_weights if len(layer_weights) in [0, 2] and len(layer_weights) != 0 else (layer_weights[0] if len(layer_weights)==2 else None))
            biases_list.append(layer_weights[1] if len(layer_weights)==2 else None)
            norm_layer_params.append(None)
            config = layer.get_config()
            if ('activation' in layer.name.lower()) or isinstance(layer, keras.layers.Activation):
                activation = config.get('activation', 'linear')
                activation_functions.append(activation)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
            elif ('flatten' in layer.name.lower()) or isinstance(layer, keras.layers.Flatten):
                activation_functions.append('flatten')
                alphas.append(0.0)
                dropout_rates.append(0.0)
            elif ('batch_normalization' in layer.name.lower()) or isinstance(layer, keras.layers.BatchNormalization):
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    norm_layer_params[-1] = (gamma, beta, moving_mean, moving_variance, epsilon)
                    activation_functions.append('batchNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    activation_functions.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
            elif ('layer_normalization' in layer.name.lower()) or isinstance(layer, keras.layers.LayerNormalization):
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    norm_layer_params[-1] = (gamma, beta, None, None, epsilon)
                    activation_functions.append('layerNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    activation_functions.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
            else:
                if len(layer_weights) == 2:
                    weights, biases = layer_weights
                else:
                    weights, biases = None, None
                weights_list[-1] = weights
                biases_list[-1] = biases
                activation = config.get('activation', 'linear')
                activation_functions.append(activation if activation != 'linear' else 'linear')
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)
        # End of layer loop

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        activation_functions = ['leakyRelu' if act == 'LeakyRelu' else act for act in activation_functions]
        # Determine input_size using model.input_shape (expects a tuple like (None, 8, 8, 1))
        if hasattr(model, "input_shape") and model.input_shape is not None:
            import numpy as np
            input_size = int(np.prod(model.input_shape[1:]))
        elif hasattr(model.layers[0], 'input_shape'):
            in_shape = model.layers[0].input_shape
            import numpy as np
            input_size = int(np.prod(in_shape[1:]))
        else:
            raise ValueError("Unable to determine model input shape.")

    elif file_type == '.onnx':
        for initializer in model.graph.initializer:
            tensor = onnx.numpy_helper.to_array(initializer)
            if len(tensor.shape) == 2:
                weights_list.append(tensor)
            elif len(tensor.shape) == 1:
                biases_list.append(tensor)

        activation_func_map = {
            'Relu': 'relu',
            'Sigmoid': 'sigmoid',
            'Tanh': 'tanhCustom',
            'Linear': 'linear',
            'LeakyRelu': 'leakyRelu',
            'Elu': 'elu',
            'Softmax': 'softmax',
            'Swish': 'swish',
            'BatchNormalization': 'batchNormalization'
        }

        for node in model.graph.node:
            act_name = activation_func_map.get(node.op_type, 'linear')
            activation_functions.append(act_name)
            alpha_val = 0.0

            if node.op_type == "LeakyRelu":
                found_alpha = False
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha_val = attr.f
                        found_alpha = True
                        break
                if not found_alpha:
                    alpha_val = 0.01
            elif node.op_type == "Elu":
                found_alpha = False
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha_val = attr.f
                        found_alpha = True
                        break
                if not found_alpha:
                    alpha_val = 1.0

            alphas.append(alpha_val if act_name != 'linear' else 0.0)

            if node.op_type == "BatchNormalization":
                gamma, beta, mean, variance, epsilon = None, None, None, None, 1e-5
                for attr in node.attribute:
                    if attr.name == "scale":
                        gamma = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "B":
                        beta = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "mean":
                        mean = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "var":
                        variance = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "epsilon":
                        epsilon = attr.f
                norm_layer_params.append((gamma, beta, mean, variance, epsilon))
                dropout_rates.append(0.0)
            else:
                norm_layer_params.append(None)
                dropout_rates.append(0.0)

        dropout_rates = [0.0] * len(weights_list)
        input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size
