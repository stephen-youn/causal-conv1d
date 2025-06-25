#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "causal_conv1d.h"

// Declarations of HIP kernels

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, hipStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, hipStream_t stream);

static void set_conv_params_fwd(ConvParamsBase &params, int batch, int dim,
                                int seqlen, int width,
                                float *x_ptr, float *w_ptr, float *out_ptr,
                                float *bias_ptr=nullptr) {
    std::memset(&params, 0, sizeof(params));
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = false;
    params.x_ptr = x_ptr;
    params.weight_ptr = w_ptr;
    params.bias_ptr = bias_ptr;
    params.out_ptr = out_ptr;
    params.x_batch_stride = dim * seqlen;
    params.x_c_stride = seqlen;
    params.x_l_stride = 1;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.out_batch_stride = dim * seqlen;
    params.out_c_stride = seqlen;
    params.out_l_stride = 1;
}

static void set_conv_params_update(ConvParamsBase &params, int batch, int dim,
                                   int seqlen, int width, int state_len,
                                   float *x_ptr, float *state_ptr,
                                   float *w_ptr, float *out_ptr,
                                   float *bias_ptr=nullptr) {
    set_conv_params_fwd(params, batch, dim, seqlen, width,
                        x_ptr, w_ptr, out_ptr, bias_ptr);
    params.conv_state_ptr = state_ptr;
    params.conv_state_len = state_len;
    params.conv_state_batch_stride = dim * state_len;
    params.conv_state_c_stride = state_len;
    params.conv_state_l_stride = 1;
}

static void fill_random(std::vector<float> &v) {
    for (auto &x : v) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

static void conv1d_ref(int batch, int dim, int seqlen, int width,
                       const std::vector<float> &x,
                       const std::vector<float> &w,
                       const std::vector<float> &bias,
                       std::vector<float> &out) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < dim; ++c) {
            for (int t = 0; t < seqlen; ++t) {
                float acc = 0.f;
                for (int k = 0; k < width; ++k) {
                    int idx = t - k;
                    float val = idx >= 0 ? x[(b*dim + c)*seqlen + idx] : 0.f;
                    acc += val * w[c*width + k];
                }
                if (!bias.empty()) acc += bias[c];
                out[(b*dim + c)*seqlen + t] = acc;
            }
        }
    }
}

static void conv1d_update_ref(int batch, int dim, int seqlen, int width,
                              int state_len, const std::vector<float> &x,
                              std::vector<float> &state,
                              const std::vector<float> &w,
                              const std::vector<float> &bias,
                              std::vector<float> &out) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            for (int c = 0; c < dim; ++c) {
                float acc = 0.f;
                for (int k = 0; k < width - 1; ++k) {
                    float val = state[((b*dim + c)*state_len) + state_len - width + 1 + k];
                    acc += val * w[c*width + k];
                }
                float in_val = x[(b*dim + c)*seqlen + t];
                acc += in_val * w[c*width + width - 1];
                if (!bias.empty()) acc += bias[c];
                out[(b*dim + c)*seqlen + t] = acc;
            }
            for (int c = 0; c < dim; ++c) {
                float *s = state.data() + (b*dim + c)*state_len;
                std::memmove(s, s + 1, sizeof(float)*(state_len - 1));
                s[state_len - 1] = x[(b*dim + c)*seqlen + t];
            }
        }
    }
}

int main() {
    const int batch = 2;
    const int dim = 4;
    const int seqlen = 8;
    const int width = 3;
    const int state_len = width + 1;

    std::vector<float> x(batch * dim * seqlen);
    std::vector<float> weight(dim * width);
    std::vector<float> bias(dim);
    std::vector<float> conv_state(batch * dim * state_len);
    std::vector<float> out_fwd(batch * dim * seqlen, 0.f);
    std::vector<float> out_update(batch * dim * seqlen, 0.f);
    std::vector<float> ref_fwd(batch * dim * seqlen, 0.f);
    std::vector<float> ref_update(batch * dim * seqlen, 0.f);
    std::vector<float> conv_state_ref(conv_state.size());

    fill_random(x);
    fill_random(weight);
    fill_random(bias);
    fill_random(conv_state);
    conv_state_ref = conv_state;

    conv1d_ref(batch, dim, seqlen, width, x, weight, bias, ref_fwd);
    conv1d_update_ref(batch, dim, seqlen, width, state_len,
                      x, conv_state_ref, weight, bias, ref_update);

    float *d_x, *d_w, *d_bias, *d_out_fwd, *d_state, *d_out_upd;
    hipMalloc(&d_x, x.size()*sizeof(float));
    hipMalloc(&d_w, weight.size()*sizeof(float));
    hipMalloc(&d_bias, bias.size()*sizeof(float));
    hipMalloc(&d_state, conv_state.size()*sizeof(float));
    hipMalloc(&d_out_fwd, out_fwd.size()*sizeof(float));
    hipMalloc(&d_out_upd, out_update.size()*sizeof(float));

    hipMemcpy(d_x, x.data(), x.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w, weight.data(), weight.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_bias, bias.data(), bias.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_state, conv_state.data(), conv_state.size()*sizeof(float), hipMemcpyHostToDevice);

    ConvParamsBase params_fwd;
    set_conv_params_fwd(params_fwd, batch, dim, seqlen, width,
                        d_x, d_w, d_out_fwd, d_bias);
    causal_conv1d_fwd_cuda<float, float>(params_fwd, 0);
    hipDeviceSynchronize();

    ConvParamsBase params_upd;
    set_conv_params_update(params_upd, batch, dim, seqlen, width, state_len,
                           d_x, d_state, d_w, d_out_upd, d_bias);
    causal_conv1d_update_cuda<float, float>(params_upd, 0);
    hipDeviceSynchronize();

    hipMemcpy(out_fwd.data(), d_out_fwd, out_fwd.size()*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(out_update.data(), d_out_upd, out_update.size()*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(conv_state.data(), d_state, conv_state.size()*sizeof(float), hipMemcpyDeviceToHost);

    double max_err_fwd = 0., max_err_update = 0.;
    for (size_t i = 0; i < out_fwd.size(); ++i) {
        max_err_fwd = std::max(max_err_fwd, std::abs(out_fwd[i] - ref_fwd[i]));
    }
    for (size_t i = 0; i < out_update.size(); ++i) {
        max_err_update = std::max(max_err_update, std::abs(out_update[i] - ref_update[i]));
    }

    std::cout << "Max error forward: " << max_err_fwd << "\n";
    std::cout << "Max error update: " << max_err_update << "\n";

    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_bias);
    hipFree(d_state);
    hipFree(d_out_fwd);
    hipFree(d_out_upd);
    return (max_err_fwd < 1e-3 && max_err_update < 1e-3) ? 0 : 1;
}

