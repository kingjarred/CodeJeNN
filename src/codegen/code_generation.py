import os

# --- OLD CODE (Preamble Header) ---
def preambleHeader():
    """
    Generate a general preamble for header file
    """
    cpp_code = """#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

// Added for convolution and pooling functions
#include <limits>

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 
"""
    return cpp_code

# # --- OLD CODE (Original code generation) ---
# """
# def codeGen(cpp_code, cpp_lambda, precision_type, weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size, user_file, input_norms, input_mins, output_norms, output_mins):
#     activation_func_map = {
#         'relu': 'relu',
#         'sigmoid': 'sigmoid',
#         'tanhCustom': 'tanhCustom',
#         'linear': 'linear',
#         'leakyRelu': 'leakyRelu',
#         'elu': 'elu',
#         'softmax': 'linear',  
#         'selu': 'selu',
#         'swish': 'swish',
#         'silu': 'silu',
#         'batchNormalization': None,
#         'flatten': None,
#         'convolutionalLayer': None
#     }
#     name_space = os.path.splitext(os.path.basename(user_file))[0]
#     name_space = name_space.replace("-", "_").replace(" ", "_")
#     cpp_code += f"""
# template <typename Scalar = {precision_type}>
# auto {name_space}(const std::array<Scalar, {input_size}>& initial_input) {{ 
# """
#     if input_norms is not None:
#         cpp_code += f"    constexpr std::array<Scalar, {len(input_norms)}> input_norms = {{"
#         cpp_code += ", ".join(f"{x:10.9e}" for x in input_norms)
#         cpp_code += "};\n\n"
#         cpp_code += f"    constexpr std::array<Scalar, {len(input_mins)}> input_mins = {{"
#         cpp_code += ", ".join(f"{x:10.9e}" for x in input_mins)
#         cpp_code += "};\n\n"
#         cpp_code += f"""    std::array<Scalar, {input_size}> model_input;
#     for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]); }}
#     if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }} 
# """
#     else:
#         cpp_code += f"    std::array<Scalar, {input_size}> model_input = initial_input;\n\n"
#         cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'
#     cpp_code += """    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 
# \n"""
#     # ... (Original dense layer propagation code follows)
#     return cpp_code
# """

# --- NEW CODE (Updated to support convolutional and pooling layers) ---
def codeGen(cpp_code, cpp_lambda, precision_type, weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size, user_file, input_norms, input_mins, output_norms, output_mins):
    activation_func_map = {
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanhCustom': 'tanhCustom',
        'linear': 'linear',
        'leakyRelu': 'leakyRelu',
        'elu': 'elu',
        'softmax': 'linear',  
        'selu': 'selu',
        'swish': 'swish',
        'silu': 'silu',
        'batchNormalization': None,
        'flatten': None,
        # convolutional layers are handled separately below
    }

    name_space = os.path.splitext(os.path.basename(user_file))[0]
    name_space = name_space.replace("-", "_").replace(" ", "_")

    cpp_code += f"""
template <typename Scalar = {precision_type}>
auto {name_space}(const std::array<Scalar, {input_size}>& initial_input) {{ 
"""
    # Input normalization
    if input_norms is not None:
        cpp_code += f"    constexpr std::array<Scalar, {len(input_norms)}> input_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_norms)
        cpp_code += "};\n\n"
        cpp_code += f"    constexpr std::array<Scalar, {len(input_mins)}> input_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_mins)
        cpp_code += "};\n\n"
        cpp_code += f"""    std::array<Scalar, {input_size}> model_input;
    for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]); }}
    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }} 
"""
    else:
        cpp_code += f"    std::array<Scalar, {input_size}> model_input = initial_input;\n\n"
        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    cpp_code += """    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 
\n"""

    # Generate constexpr arrays for Dense layer weights, biases, and normalization parameters (if any)
    for i, (w, b, norm_params, conv_params) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params)):
        layer_index = i + 1
        # For dense (fully-connected) layers: weights and biases are provided.
        if w is not None and b is not None and conv_params is None:
            weights_flat = w.flatten()
            biases_flat = b.flatten()
            cpp_code += f"    constexpr std::array<Scalar, {len(weights_flat)}> weights_{layer_index} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in weights_flat)
            cpp_code += "};\n\n"
            cpp_code += f"    constexpr std::array<Scalar, {len(biases_flat)}> biases_{layer_index} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in biases_flat)
            cpp_code += "};\n\n"
        # For convolution or pooling layers, we assume parameters are stored inside conv_params.
        elif conv_params is not None:
            conv_type = conv_params.get("layer_type", "")
            cpp_code += f"    // Layer {layer_index} ({conv_type}) shapes:\n"
            if "input_shape" in conv_params and conv_params["input_shape"] is not None:
                in_shape = conv_params["input_shape"][1:]
                cpp_code += f"    constexpr std::array<int, {len(in_shape)}> layer_{layer_index}_input_shape = {{{', '.join(str(x) for x in in_shape)}}};\n"
            else:
                raise ValueError(f"Missing input_shape for convolution/pooling layer {layer_index} of type {conv_type}")
            if "output_shape" in conv_params and conv_params["output_shape"] is not None:
                out_shape = conv_params["output_shape"][1:]
                cpp_code += f"    constexpr std::array<int, {len(out_shape)}> layer_{layer_index}_output_shape = {{{', '.join(str(x) for x in out_shape)}}};\n"
            if "strides" in conv_params and conv_params["strides"] is not None:
                strides = conv_params["strides"]
                cpp_code += f"    constexpr std::array<int, {len(strides)}> layer_{layer_index}_strides = {{{', '.join(str(x) for x in strides)}}};\n"
            if "dilation_rate" in conv_params and conv_params["dilation_rate"] is not None:
                dilation = conv_params["dilation_rate"]
                cpp_code += f"    constexpr std::array<int, {len(dilation)}> layer_{layer_index}_dilation_rate = {{{', '.join(str(x) for x in dilation)}}};\n"
            if "kernel_shape" in conv_params and conv_params["kernel_shape"] is not None:
                k_shape = conv_params["kernel_shape"]
                cpp_code += f"    constexpr std::array<int, {len(k_shape)}> layer_{layer_index}_kernel_shape = {{{', '.join(str(x) for x in k_shape)}}};\n"
            if conv_type == "SeparableConv2D":
                if conv_params.get("depthwise_kernel_shape") is not None:
                    dws = conv_params["depthwise_kernel_shape"]
                    cpp_code += f"    constexpr std::array<int, {len(dws)}> layer_{layer_index}_depthwise_kernel_shape = {{{', '.join(str(x) for x in dws)}}};\n"
                if conv_params.get("pointwise_kernel_shape") is not None:
                    pws = conv_params["pointwise_kernel_shape"]
                    cpp_code += f"    constexpr std::array<int, {len(pws)}> layer_{layer_index}_pointwise_kernel_shape = {{{', '.join(str(x) for x in pws)}}};\n"
            cpp_code += "\n"
        # For normalization layers (dense only)
        if norm_params is not None:
            gamma, beta, mean, variance, epsilon = norm_params
            if gamma is not None:
                gamma_flat = gamma.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(gamma_flat)}> gamma_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in gamma_flat)
                cpp_code += "};\n\n"
            if beta is not None:
                beta_flat = beta.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(beta_flat)}> beta_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in beta_flat)
                cpp_code += "};\n\n"
            if mean is not None:
                mean_flat = mean.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(mean_flat)}> mean_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in mean_flat)
                cpp_code += "};\n\n"
            if variance is not None:
                variance_flat = variance.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(variance_flat)}> variance_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in variance_flat)
                cpp_code += "};\n\n"
            cpp_code += f"    constexpr Scalar epsilon_{layer_index} = {epsilon:10.9e};\n\n"

    cpp_code += cpp_lambda
    cpp_code += """\n    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 
\n"""

    last_layer = "model_input"
    last_size = input_size

    # Propagation: iterate over layers and generate function calls.
    for i, (w, b, norm_params, conv_params, act_func, alpha) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params, activation_functions, alphas)):
        layer_index = i + 1
        # For Dense layers (non-convolutional)
        if conv_params is None:
            if act_func == 'tanh':
                act_func = 'tanhCustom'
            mapped_act = activation_func_map.get(act_func, 'linear')
            if w is not None and b is not None:
                output_size = w.shape[1]
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
                cpp_code += f"    forwardPass<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), weights_{layer_index}.data(), biases_{layer_index}.data(), {last_size}, {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_index}_output"
                last_size = output_size
            elif act_func == 'batchNormalization' and norm_params is not None:
                output_size = len(norm_params[0])
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
                cpp_code += f"    batchNormalization<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), gamma_{layer_index}.data(), beta_{layer_index}.data(), mean_{layer_index}.data(), variance_{layer_index}.data(), epsilon_{layer_index});\n\n"
                last_layer = f"layer_{layer_index}_output"
                last_size = output_size
            elif act_func == 'layerNormalization' and norm_params is not None:
                output_size = len(norm_params[0])
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
                cpp_code += f"    layerNormalization<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), gamma_{layer_index}.data(), beta_{layer_index}.data(), epsilon_{layer_index});\n\n"
                last_layer = f"layer_{layer_index}_output"
                last_size = output_size
            elif act_func == 'flatten':
                output_size = last_size
                cpp_code += f"    // Flatten layer not explicitly handled, assuming no-op\n"
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output = {last_layer};\n\n"
                last_layer = f"layer_{layer_index}_output"
            elif w is None and b is None and norm_params is None and act_func is not None:
                output_size = last_size
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
                for idx in range(output_size):
                    cpp_code += f"    {mapped_act}(layer_{layer_index}_output[{idx}], {last_layer}[{idx}], {alpha});\n"
                cpp_code += "\n"
                last_layer = f"layer_{layer_index}_output"
                last_size = output_size
        else:
            # Branch for Convolutional and Pooling layers
            conv_type = conv_params.get("layer_type", "")
            if conv_type in ["Conv2D", "Conv2DTranspose"]:
                if conv_params.get("input_shape") is None:
                    raise ValueError(f"Missing input_shape for convolution layer {layer_index} of type {conv_type}")
                padding = conv_params.get("padding", "valid")
                H_in, W_in, C_in = conv_params["input_shape"][1:4]
                H_k, W_k, _, C_out = conv_params["kernel_shape"]
                StrideH, StrideW = conv_params.get("strides", (1,1))
                out_shape = conv_params["output_shape"][1:]
                out_size = 1
                for dim in out_shape:
                    out_size *= dim
                cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_index}_output;\n"
                func_name = "conv2D_valid" if padding=="valid" else "conv2D_same"
                cpp_code += f"    {func_name}<Scalar, {H_in}, {W_in}, {C_in}, {H_k}, {W_k}, {C_out}, {StrideH}, {StrideW}>( {last_layer}.data(), weights_{layer_index}.data(), biases_{layer_index}.data(), layer_{layer_index}_output.data() );\n\n"
                last_layer = f"layer_{layer_index}_output";
                last_size = out_size;
            elif conv_type == "DepthwiseConv2D":
                if conv_params.get("input_shape") is None:
                    raise ValueError(f"Missing input_shape for DepthwiseConv2D layer {layer_index}")
                padding = conv_params.get("padding", "valid")
                H_in, W_in, C_in = conv_params["input_shape"][1:4]
                H_k, W_k, _, depth_multiplier = conv_params["kernel_shape"]
                StrideH, StrideW = conv_params.get("strides", (1,1))
                out_shape = conv_params["output_shape"][1:]
                out_size = 1
                for dim in out_shape:
                    out_size *= dim
                cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_index}_output;\n"
                func_name = "depthwiseConv2D_valid" if padding=="valid" else "depthwiseConv2D_same"
                cpp_code += f"    {func_name}<Scalar, {H_in}, {W_in}, {C_in}, {H_k}, {W_k}, {depth_multiplier}, {StrideH}, {StrideW}>( {last_layer}.data(), weights_{layer_index}.data(), biases_{layer_index}.data(), layer_{layer_index}_output.data() );\n\n"
                last_layer = f"layer_{layer_index}_output";
                last_size = out_size;
            elif conv_type == "SeparableConv2D":
                if conv_params.get("input_shape") is None:
                    raise ValueError(f"Missing input_shape for SeparableConv2D layer {layer_index}")
                padding = conv_params.get("padding", "valid")
                H_in, W_in, C_in = conv_params["input_shape"][1:4]
                H_k, W_k, _, _ = conv_params["depthwise_kernel_shape"]
                depth_multiplier = conv_params["depthwise_kernel_shape"][-1]
                _, _, _, C_out = conv_params["pointwise_kernel_shape"]
                StrideH, StrideW = conv_params.get("strides", (1,1))
                out_shape = conv_params["output_shape"][1:]
                out_size = 1
                for dim in out_shape:
                    out_size *= dim
                cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_index}_output;\n"
                func_name = "separableConv2D_valid"  # Using the VALID version for simplicity
                cpp_code += f"    {func_name}<Scalar, {H_in}, {W_in}, {C_in}, {H_k}, {W_k}, {depth_multiplier}, {C_out}, {StrideH}, {StrideW}>( {last_layer}.data(), weights_{layer_index}_depthwise.data(), weights_{layer_index}_pointwise.data(), biases_{layer_index}.data(), layer_{layer_index}_output.data() );\n\n"
                last_layer = f"layer_{layer_index}_output";
                last_size = out_size;
            elif conv_type in ["MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D", "GlobalMaxPooling2D"]:
                pool_type = conv_type
                out_shape = conv_params["output_shape"][1:]
                out_size = 1
                for dim in out_shape:
                    out_size *= dim
                cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_index}_output;\n"
                if pool_type == "MaxPooling2D":
                    pool_h, pool_w = conv_params.get("pool_size", (2,2))
                    StrideH, StrideW = conv_params.get("strides", (pool_h, pool_w))
                    H_in, W_in, C = conv_params["input_shape"][1:4]
                    cpp_code += f"    maxPooling2D<Scalar, {H_in}, {W_in}, {C}>( {last_layer}.data(), {pool_h}, {pool_w}, {StrideH}, {StrideW}, layer_{layer_index}_output.data() );\n\n"
                elif pool_type == "AveragePooling2D":
                    pool_h, pool_w = conv_params.get("pool_size", (2,2))
                    StrideH, StrideW = conv_params.get("strides", (pool_h, pool_w))
                    H_in, W_in, C = conv_params["input_shape"][1:4]
                    cpp_code += f"    averagePooling2D<Scalar, {H_in}, {W_in}, {C}>( {last_layer}.data(), {pool_h}, {pool_w}, {StrideH}, {StrideW}, layer_{layer_index}_output.data() );\n\n"
                elif pool_type == "GlobalAveragePooling2D":
                    H_in, W_in, C = conv_params["input_shape"][1:4]
                    cpp_code += f"    globalAveragePooling2D<Scalar, {H_in}, {W_in}, {C}>( {last_layer}.data(), layer_{layer_index}_output.data() );\n\n"
                elif pool_type == "GlobalMaxPooling2D":
                    H_in, W_in, C = conv_params["input_shape"][1:4]
                    cpp_code += f"    globalMaxPooling2D<Scalar, {H_in}, {W_in}, {C}>( {last_layer}.data(), layer_{layer_index}_output.data() );\n\n"
                last_layer = f"layer_{layer_index}_output";
                last_size = out_size;
            else:
                cpp_code += f"    // Unknown convolution/pooling layer type for layer {layer_index}; passing through unchanged.\n"
                cpp_code += f"    std::array<Scalar, {last_size}> layer_{layer_index}_output = {last_layer};\n\n"
                last_layer = f"layer_{layer_index}_output";
    # End for loop

    # Output un-normalization (if provided)
    if output_norms is not None:
        cpp_code += f"    constexpr std::array<Scalar, {len(output_norms)}> output_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in output_norms)
        cpp_code += "};\n\n"
        cpp_code += f"    constexpr std::array<Scalar, {len(output_mins)}> output_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in output_mins)
        cpp_code += "};\n\n"
        cpp_code += f"    std::array<Scalar, {last_size}> model_output;\n"
        cpp_code += f"    for (int i = 0; i < {last_size}; i++) {{ model_output[i] = ({last_layer}[i] * output_norms[i]) + output_mins[i]; }}\n"
    else:
        cpp_code += f"    std::array<Scalar, {last_size}> model_output = {last_layer};\n\n"

    cpp_code += f"    return model_output;\n}}\n"

    return cpp_code
