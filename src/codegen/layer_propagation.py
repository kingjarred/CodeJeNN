import os
import absl.logging
import warnings
import math
import stdexcept  # For std::numeric_limits equivalent in C++ comments
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def activationFunctions(cpp_code, activation_functions):
    """
    generate C++ lambda-based activation functions (with no indentation for the lambdas)
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

    current_activations = set(activation_functions)
    current_activations = {('tanhCustom' if act == 'tanh' else act) for act in current_activations if act is not None}

    cpp_lambda = """    //\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n"""

    for act in current_activations:
        if act in lambda_defs:
            cpp_lambda += lambda_defs[act]

    cpp_code += layerNormalization
    cpp_code += batchNormalization
    cpp_code += forwardPass

    # ===== NEW CODE: Convolution and Pooling Functions =====
    # The following functions implement basic versions of convolution and pooling.
    # They assume fixed (compile-time) dimensions and valid padding for simplicity.
    # More advanced versions (e.g. for 'same' padding, dilations, etc.) can be added similarly.

    convolution_functions = r"""
// --- Convolution Functions ---

// 2D Convolution with VALID padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int C_out, int StrideH, int StrideW>
void conv2D_valid(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int H_out = (H_in - H_k) / StrideH + 1;
    constexpr int W_out = (W_in - W_k) / StrideW + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                Scalar sum = 0;
                for (int kh = 0; kh < H_k; ++kh) {
                    for (int kw = 0; kw < W_k; ++kw) {
                        for (int cin = 0; cin < C_in; ++cin) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * C_out + c;
                            int input_index = ((h * StrideH + kh) * W_in + (w * StrideW + kw)) * C_in + cin;
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                }
                output[(h * W_out + w) * C_out + c] = sum + bias[c];
            }
        }
    }
}

// 2D Convolution with SAME padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int C_out, int StrideH, int StrideW>
void conv2D_same(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int pad_h = ((H_in - 1) * StrideH + H_k - H_in) / 2;
    constexpr int pad_w = ((W_in - 1) * StrideW + W_k - W_in) / 2;
    constexpr int H_out = (H_in + 2 * pad_h - H_k) / StrideH + 1;
    constexpr int W_out = (W_in + 2 * pad_w - W_k) / StrideW + 1;
    Scalar padded_input[(H_in + 2 * pad_h) * (W_in + 2 * pad_w) * C_in] = {0};
    for (int h = 0; h < H_in; ++h) {
        for (int w = 0; w < W_in; ++w) {
            for (int c = 0; c < C_in; ++c) {
                int in_index = (h * W_in + w) * C_in + c;
                int pad_index = ((h + pad_h) * (W_in + 2 * pad_w) + (w + pad_w)) * C_in + c;
                padded_input[pad_index] = input[in_index];
            }
        }
    }
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                Scalar sum = 0;
                for (int kh = 0; kh < H_k; ++kh) {
                    for (int kw = 0; kw < W_k; ++kw) {
                        for (int cin = 0; cin < C_in; ++cin) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * C_out + c;
                            int pad_index = ((h * StrideH + kh) * (W_in + 2 * pad_w) + (w * StrideW + kw)) * C_in + cin;
                            sum += padded_input[pad_index] * kernel[kernel_index];
                        }
                    }
                }
                output[(h * W_out + w) * C_out + c] = sum + bias[c];
            }
        }
    }
}

// Depthwise 2D Convolution with VALID padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int StrideH, int StrideW>
void depthwiseConv2D_valid(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int C_out = C_in * depth_multiplier;
    constexpr int H_out = (H_in - H_k) / StrideH + 1;
    constexpr int W_out = (W_in - W_k) / StrideW + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int cin = 0; cin < C_in; ++cin) {
                for (int m = 0; m < depth_multiplier; ++m) {
                    Scalar sum = 0;
                    for (int kh = 0; kh < H_k; ++kh) {
                        for (int kw = 0; kw < W_k; ++kw) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * depth_multiplier + m;
                            int input_index = ((h * StrideH + kh) * W_in + (w * StrideW + kw)) * C_in + cin;
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                    output[((h * W_out + w) * C_in + cin) * depth_multiplier + m] = sum + bias[cin * depth_multiplier + m];
                }
            }
        }
    }
}

// Depthwise 2D Convolution with SAME padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int StrideH, int StrideW>
void depthwiseConv2D_same(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int pad_h = ((H_in - 1) * StrideH + H_k - H_in) / 2;
    constexpr int pad_w = ((W_in - 1) * StrideW + W_k - W_in) / 2;
    constexpr int H_out = (H_in + 2 * pad_h - H_k) / StrideH + 1;
    constexpr int W_out = (W_in + 2 * pad_w - W_k) / StrideW + 1;
    Scalar padded_input[(H_in + 2 * pad_h) * (W_in + 2 * pad_w) * C_in] = {0};
    for (int h = 0; h < H_in; ++h) {
        for (int w = 0; w < W_in; ++w) {
            for (int c = 0; c < C_in; ++c) {
                int in_index = (h * W_in + w) * C_in + c;
                int pad_index = ((h + pad_h) * (W_in + 2 * pad_w) + (w + pad_w)) * C_in + c;
                padded_input[pad_index] = input[in_index];
            }
        }
    }
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int cin = 0; cin < C_in; ++cin) {
                for (int m = 0; m < depth_multiplier; ++m) {
                    Scalar sum = 0;
                    for (int kh = 0; kh < H_k; ++kh) {
                        for (int kw = 0; kw < W_k; ++kw) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * depth_multiplier + m;
                            int pad_index = ((h * StrideH + kh) * (W_in + 2 * pad_w) + (w * StrideW + kw)) * C_in + cin;
                            sum += padded_input[pad_index] * kernel[kernel_index];
                        }
                    }
                    output[((h * W_out + w) * C_in + cin) * depth_multiplier + m] = sum + bias[cin * depth_multiplier + m];
                }
            }
        }
    }
}

// SeparableConv2D (VALID padding): performs depthwise then pointwise convolution
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int C_out, int StrideH, int StrideW>
void separableConv2D_valid(const Scalar* input, const Scalar* depthwise_kernel, const Scalar* pointwise_kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int H_dw = (H_in - H_k) / StrideH + 1;
    constexpr int W_dw = (W_in - W_k) / StrideW + 1;
    constexpr int C_dw = C_in * depth_multiplier;
    Scalar depthwise_output[H_dw * W_dw * C_dw] = {0};
    depthwiseConv2D_valid<Scalar, H_in, W_in, C_in, H_k, W_k, depth_multiplier, StrideH, StrideW>(input, depthwise_kernel, nullptr, depthwise_output);
    // Pointwise convolution: kernel shape [1, 1, C_dw, C_out]
    for (int i = 0; i < H_dw * W_dw; ++i) {
        for (int c = 0; c < C_out; ++c) {
            Scalar sum = 0;
            for (int k = 0; k < C_dw; ++k) {
                sum += depthwise_output[i * C_dw + k] * pointwise_kernel[k * C_out + c];
            }
            output[i * C_out + c] = sum + bias[c];
        }
    }
}

// --- Pooling Functions ---

template<typename Scalar, int H_in, int W_in, int C>
void maxPooling2D(const Scalar* input, int pool_h, int pool_w, int stride_h, int stride_w, Scalar* output) noexcept {
    int H_out = (H_in - pool_h) / stride_h + 1;
    int W_out = (W_in - pool_w) / stride_w + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C; ++c) {
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_h; ++ph) {
                    for (int pw = 0; pw < pool_w; ++pw) {
                        int in_index = ((h * stride_h + ph) * W_in + (w * stride_w + pw)) * C + c;
                        if (input[in_index] > max_val)
                            max_val = input[in_index];
                    }
                }
                output[(h * W_out + w) * C + c] = max_val;
            }
        }
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void averagePooling2D(const Scalar* input, int pool_h, int pool_w, int stride_h, int stride_w, Scalar* output) noexcept {
    int H_out = (H_in - pool_h) / stride_h + 1;
    int W_out = (W_in - pool_w) / stride_w + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C; ++c) {
                Scalar sum = 0;
                for (int ph = 0; ph < pool_h; ++ph) {
                    for (int pw = 0; pw < pool_w; ++pw) {
                        int in_index = ((h * stride_h + ph) * W_in + (w * stride_w + pw)) * C + c;
                        sum += input[in_index];
                    }
                }
                output[(h * W_out + w) * C + c] = sum / (pool_h * pool_w);
            }
        }
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void globalAveragePooling2D(const Scalar* input, Scalar* output) noexcept {
    int size = H_in * W_in;
    for (int c = 0; c < C; ++c) {
        Scalar sum = 0;
        for (int i = 0; i < size; ++i) {
            sum += input[i * C + c];
        }
        output[c] = sum / size;
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void globalMaxPooling2D(const Scalar* input, Scalar* output) noexcept {
    int size = H_in * W_in;
    for (int c = 0; c < C; ++c) {
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int i = 0; i < size; ++i) {
            if (input[i * C + c] > max_val)
                max_val = input[i * C + c];
        }
        output[c] = max_val;
    }
}
"""
    cpp_code += convolution_functions

    return cpp_code, cpp_lambda
