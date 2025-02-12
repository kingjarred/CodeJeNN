import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def activationFunctions(cpp_code, activation_functions):
    """
    Generate C++ lambda-based activation functions (with no indentation for the lambdas)
    and normalization functions. ForwardPass also remains as Code 2 style.
    """
    lambda_defs = {
        'relu': """
    auto relu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };
""",
        'sigmoid': """
    auto sigmoid = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };
""",
        'tanhCustom': """
    auto tanhCustom = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };
""",
        'leakyRelu': """
    // OLD CODE:
    // auto leakyRelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
    //     output = input > 0 ? input : alpha * input;
    // };
    // NEW CODE:
    auto leakyRelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        'linear': """
    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };
""",
        'elu': """
    auto elu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };
""",
        'selu': """
    template<typename T> constexpr T SELU_LAMBDA = static_cast<T>(1.0507009873554804934193349852946);
    template<typename T> constexpr T SELU_ALPHA = static_cast<T>(1.6732632423543772848170429916717);
    auto selu = [](Scalar& output, Scalar input, Scalar alpha = SELU_ALPHA<double>) noexcept {
        using Scalar = decltype(input);
        output = SELU_LAMBDA<Scalar> * (input > 0 ? input : alpha * (std::exp(input) - 1));
    };
""",
        'swish': """
    auto swish = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input / (1 + std::exp(-alpha * input));
    };
""",
        'prelu': """
    auto prelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        'silu': """
    auto silu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };
"""
    }

    layerNormalization = """
template<typename Scalar, int size>
void layerNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, Scalar epsilon) noexcept {
    Scalar mean = 0;
    Scalar variance = 0;
    for (int i = 0; i < size; ++i) {
        mean += inputs[i];
    }
    mean /= size;
    for (int i = 0; i < size; ++i) {
        variance += (inputs[i] - mean) * (inputs[i] - mean);
    }
    variance /= size;
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
    }
}
"""

    batchNormalization = """
template<typename Scalar, int size>
void batchNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
"""

    forwardPass = """
template<typename Scalar, int output_size>
void forwardPass(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for(int i = 0; i < output_size; ++i){
        Scalar sum = 0;
        for(int j = 0; j < input_size; ++j){
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}
"""

    # // OLD CODE: No convolution functions existed previously.
    # // NEW CODE: Add convolution function definitions

    conv2DForward = """
template<typename Scalar, int out_channels, int out_height, int out_width>
void conv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_channels, int in_height, int in_width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int in_h = oh * stride_h - pad_h + kh;
                            int in_w = ow * stride_w - pad_w + kw;
                            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + ic;
                                int weight_index = (((kh * kernel_w + kw) * in_channels + ic) * out_channels) + oc;
                                sum += inputs[input_index] * weights[weight_index];
                            }
                        }
                    }
                }
                sum += biases[oc];
                activation_function(outputs[(oh * out_width * out_channels) + (ow * out_channels) + oc], sum, alpha);
            }
        }
    }
}
"""

    conv2DTransposeForward = """
template<typename Scalar, int out_channels, int out_height, int out_width>
void conv2DTransposeForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified implementation for transposed convolution (stub)
    for (int i = 0; i < out_height * out_width * out_channels; ++i) {
        outputs[i] = 0;
    }
    // ... (proper transposed convolution implementation would go here)
    for (int i = 0; i < out_height * out_width * out_channels; ++i) {
        activation_function(outputs[i], outputs[i], alpha);
    }
}
"""

    conv1DForward = """
template<typename Scalar, int out_size>
void conv1DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_size, int kernel_size, int stride, int pad,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for (int o = 0; o < out_size; ++o) {
        Scalar sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int in_index = o * stride - pad + k;
            if(in_index >= 0 && in_index < in_size){
                int weight_index = k * out_size + o;
                sum += inputs[in_index] * weights[weight_index];
            }
        }
        sum += biases[o];
        activation_function(outputs[o], sum, alpha);
    }
}
"""

    conv3DForward = """
template<typename Scalar, int out_channels, int out_depth, int out_height, int out_width>
void conv3DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified 3D convolution implementation
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int od = 0; od < out_depth; ++od) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    Scalar sum = 0;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = 0; kd < kernel_d; ++kd) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int in_d = od * stride_d - pad_d + kd;
                                    int in_h = oh * stride_h - pad_h + kh;
                                    int in_w = ow * stride_w - pad_w + kw;
                                    if(in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width){
                                        int input_index = ((in_d * in_height * in_width * in_channels) + (in_h * in_width * in_channels) + (in_w * in_channels) + ic);
                                        int weight_index = (((((kd * kernel_h + kh) * kernel_w + kw) * in_channels + ic) * out_channels) + oc);
                                        sum += inputs[input_index] * weights[weight_index];
                                    }
                                }
                            }
                        }
                    }
                    sum += biases[oc];
                    int output_index = ((od * out_height * out_width * out_channels) + (oh * out_width * out_channels) + (ow * out_channels) + oc);
                    activation_function(outputs[output_index], sum, alpha);
                }
            }
        }
    }
}
"""

    depthwiseConv2DForward = """
template<typename Scalar, int out_channels, int out_height, int out_width>
void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified depthwise convolution implementation (each input channel is convolved independently)
    for (int c = 0; c < in_channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = oh * stride_h - pad_h + kh;
                        int in_w = ow * stride_w - pad_w + kw;
                        if(in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width){
                            int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
                            int weight_index = (kh * kernel_w + kw) * in_channels + c;
                            sum += inputs[input_index] * weights[weight_index];
                        }
                    }
                }
                sum += biases[c];
                int output_index = (oh * out_width * in_channels) + (ow * in_channels) + c;
                activation_function(outputs[output_index], sum, alpha);
            }
        }
    }
}
"""

    separableConv2DForward = """
template<typename Scalar, int out_channels, int out_height, int out_width>
void separableConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* depthwise_weights, const Scalar* pointwise_weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // First perform depthwise convolution (this is a simplified approach)
    const int depthwise_output_size = in_height * in_width * in_channels; // assuming same spatial dims for simplicity
    Scalar depthwise_output[depthwise_output_size];
    depthwiseConv2DForward<Scalar, in_channels, in_height, in_width>(depthwise_output, inputs, depthwise_weights, biases, in_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, linear, 0.0);
    // Then perform pointwise convolution
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < in_height * in_width; ++i) {
            Scalar sum = 0;
            for (int ic = 0; ic < in_channels; ++ic) {
                int index = i * in_channels + ic;
                int weight_index = ic * out_channels + oc;
                sum += depthwise_output[index] * pointwise_weights[weight_index];
            }
            sum += biases[oc];
            outputs[i * out_channels + oc] = sum;
            activation_function(outputs[i * out_channels + oc], sum, alpha);
        }
    }
}
"""

    convLSTM2DForward = """
template<typename Scalar>
void convLSTM2DForward(/* parameters */) noexcept {
    // Stub for ConvLSTM2D.
    // A full implementation would require handling time steps and cell states.
}
"""

    current_activations = set(activation_functions)
    current_activations = {('tanhCustom' if act == 'tanh' else act) for act in current_activations if act is not None}

    cpp_lambda = """    //\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n"""

    for act in current_activations:
        if act in lambda_defs:
            cpp_lambda += lambda_defs[act]
            
    # // Append the old normalization and forward functions first
    cpp_code += layerNormalization
    cpp_code += batchNormalization
    cpp_code += forwardPass
    # // OLD CODE: (No conv functions existed previously.)
    # // NEW CODE: Append our newly added convolution functions
    cpp_code += conv2DForward
    cpp_code += conv2DTransposeForward
    cpp_code += conv1DForward
    cpp_code += conv3DForward
    cpp_code += depthwiseConv2DForward
    cpp_code += separableConv2DForward
    cpp_code += convLSTM2DForward

    return cpp_code, lambda_defs
