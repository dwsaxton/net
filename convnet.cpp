#include "convnet.h"
#include "cube.h"

#include <cassert>
#include <iostream>
using namespace std;

const float MOMENTUM = 0.9;

ConvNet::ConvNet() {
}

ConvNet::ConvNet(vector<LayerParams> const& params) {
  layer_count_ = params.size();
  assert(layer_count_ >= 1);
  
  params_ = params;
  layers_.resize(layer_count_);
  
  for (int i = layer_count - 1; i >= 0; --i) {
    layers_[i].value = Cube(params[i].features, params[i].edge, params[i].edge);
    layers_[i].value_deriv = layers_[i].value;
    
    if (i > 0) {
      int input_features = params[i - 1].features;
      layers_[i].kernels.resize(input_features);
      for (int j = 0; j < input_features; ++j) {
        layers_[i].kernels[j] = Kernel(input_features, params[i].kernel, params[i].kernel);
        layers_[i].kernels_deriv[j] = layers_[i].kernels[j];
      }
    }
    
    layers_[i].randomizeKernels();
    
    // And do some sanity checking
    LayerParams::ConnectionType type = params[i].connection_type;
    if (i == 0) {
      assert(type == LayerParams::Initial);
      assert(params[i].box_count == 1);
    } else if (type == LayerParams::Convolution) {
      assert(params[i].mask_edge % 2 == 1);
      int mid = (params[i].mask_edge - 1) / 2;
      assert(params[i].box_edge == params[i - 1].box_edge - 2 * mid);
    } else {
      assert(type == LayerParams::SoftMax);
      assert(params[i].box_count == params[i - 1].box_count);
      assert(params[i].box_edge == params[i - 1].box_edge);
    }
  }
}

void softMax(Cube const& input, Cube &output) {
  assert(input.rows() == output.rows());
  assert(input.cols() == output.cols());
  assert(input.height() == output.height());
  assert(input.height() == 1);
  
  MatrixXf const& input_data = input.layer(0);
  MatrixXf & output_data = output.layer(0);
  
  // use max_value to divide the top and bottom of the fraction in SoftMax by exp(max_value),
  // to avoid float-overflow
  float max_value = input_data.maxCoeff();
  
  output_data = (input_data.array() - max_value).exp().matrix();
  output_data /= output_data.sum();
}

void softMaxBack(Cube const& output_cube_value, Cube const& output_cube_deriv, Cube & input_cube_deriv) {
  assert(second_edge == 1); // assuming each box is a "singleton" box
  assert(first_edge == 1); // similarly for layer below
  
  MatrixXf const& output_value = output_cube_value.layer(0);
  MatrixXf const& output_deriv = output_cube_deriv.layer(0);
  MatrixXf & input_deriv = input_cube_deriv.layer(0);
  
  input_deriv = -output_value.cwiseProduct(output_deriv).sum() * output_value + output_value.cwiseProduct(output_deriv);
}

const float leak = 0.0001;

float relu(float x) {
  return x > 0 ? x : leak * x;
}

void convolution(Cube const& input, vector<Kernel> const& masks, int stride, Cube &output) {
  int output_features = masks.size();
  assert(output.d0() == output_features);
  int edge = masks[0].kernel().d1();
  assert(edge == masks[0].kernel().d2()); // only support square masks
  assert(edge % 2 == 1); // only support odd-sized masks
  int mid = (edge - 1) / 2;
  assert(masks[0].kernel().d0() == input.d0());
  
  assert(input.d1() == input.d2()); // only support square inputs
  assert(output.d1() == output.d2()); // only support square outputs
  
  assert((input.d1() - 1) % stride == 0);
  assert(output.d1() == (input.d1() - 1) / stride + 1);
  
  for (int i = 0; i < output_features; ++i) {
    for (int j = 0; j < output.d1(); ++j) {
      for (int k = 0; k < output.d2(); ++k) {
        int in_x = j * stride;
        int in_y = k * stride;
        float sum = masks[i].bias + input.computeKernel(masks[i].kernel, in_x, in_y);
        output(i, j, k) = relu(sum);
      }
    }
  }
}

void convolutionBack(Cube const& output_value, Cube const& output_deriv, int stride, Cube const& input_value, Cube & input_deriv, vector<Kernel> & mask_deriv) {
  int output_features = mask_deriv.size();
  assert(output_features == output_value.d0());
  
  input_deriv.setZero();
  for (int i = 0; i < mask_deriv.size(); ++i) {
    mask_deriv[i].setZero();
  }
  
  for (int i = 0; i < output_features; ++i) {
    for (int j = 0; j < output_value.d1(); ++j) {
      for (int k = 0; k < output_value.d2(); ++k) {
        float deriv_base = output_deriv(i, j, k);
        if (output_value(i, j, k) < 0) {
          deriv_base *= leak;
        }
    
        int in_x = j * stride;
        int in_y = k * stride;
    
        input_deriv.block(i, in_x, in_y) += deriv_base * mask_deriv[i];
        mask_deriv[i].bias += deriv_base;
        mask_deriv[i] += deriv_base * input_value.block(0, in_x, in_y); // TODO this op doesn't exist
      }
    }
  }
}

void ConvNet::forwardPass(MatrixXf const& input) {
  int range_i = params_[0].box_edge - input.rows();
  int range_j = params_[0].box_edge - input.cols();
  assert(range_i >= 0);
  assert(range_j >= 0);
  int offset_i = range_i == 0 ? 0 : rand() % range_i;
  int offset_j = range_j == 0 ? 0 : rand() % range_j;
  if (range_i == 0 && range_j == 0) {
    layers_[0].values(0) = input;
  } else {
    layers_[0].values(0).setZero();
    layers_[0].values(0).block(offset_i, offset_j, input.rows(), input.cols()) = input;
  }
  
  for (int layer = 1; layer < layer_count_; ++layer) {
    Layer const& input = layers_[layer - 1];
    Layer & output = layers_[layer];
    
    switch (params_[layer].connection_type) {
      case LayerParams::Initial:
        // already handled
        continue;
      case LayerParams::SoftMax:
        softMax(input.values, output.values);
        continue;
      case LayerParams::Convolution:
        int stride = params_[layer + 1].connection_type == LayerParams::Pooling ? 2 : 1;
        convolution(input.values, input.masks, stride, output.values);
        continue;
    }
  }
}

void ConvNet::backwardsPass(VectorXf const& target, float learning_rate) {
  // already checked earlier, but do it here too to indicate where it gets used
  assert(params_[layer_count_ - 1].box_edge == 1);
  
  assert(params_[layer_count_ - 1].box_count == target.size());
  
  // Initialize the first set of derivatives, using the error function
  // E = (1/2) \sum_{i = 1}^{N} (v[i] - target[i])^2
  // so {dE}/{dv[i]} = v[i] - target[i]
  Layer & top_layer = layers_[layer_count_ - 1];
  for (int i = 0; i < target.size(); ++i) {
    top_layer.deriv_values(i) = top_layer.values(i, 0, 0) - target(i);
  }
    
  for (int layer = layer_count_ - 1; layer >= 1; --layer) {
    Layer &input = layers_[layer - 1];
    Layer const& output = layers_[layer];
    
    switch (params_[layer].connection_type) {
      case LayerParams::Initial:
        assert(false); // should only be initial layer for layer = 0
        continue;
      case LayerParams::SoftMax:
        softMaxBack(output.values, output.deriv_values, input.deriv_values, input.deriv_masks);
        continue;
      case LayerParams::Convolution:
        int stride = params_[layer + 1].connection_type == LayerParams::Pooling ? 2 : 1;
        convolutionBack(output.values, output.deriv_values, stride, input.values, input.deriv_values, input.deriv_masks);
        continue;
    }
  }
  
  for (int layer = 1; layer < layer_count_; ++layer) {
    if (params_[layer].connection_type == LayerParams::SoftMax) {
      continue;
    }
    int first_edge = params_[layer - 1].box_edge;
    int second_edge = params_[layer].box_edge;
    int box_count = params_[layer].box_count;
    for (int box_index = 0; box_index < box_count; ++box_index) {
      layers_[layer].boxes[box_index].weights.update(MOMENTUM, learning_rate);
    }
  }
}

VectorXf ConvNet::getOutput() const {
  MatrixXf layer = layers_[layer_count_ - 1].layers(0);
  if (layer.cols() == 1) {
    return layer.block(0, 0, layer.rows(), 1);
  } else {
    assert(layer.rows() == 1);
    return layer.block(0, 0, 1, layer.cols());
  }
}


void Layer::setDerivsZero() {
  for (int i = 0; i < boxes.size(); ++i) {
    boxes[i].deriv_values.setZero();
    boxes[i].weights.setDerivsZero();
  }
}

void Layer::randomizeKernels() {
  for (int i = 0; i < kernels.size(); ++i) {
    kernels[i].cube.setRandom();
  }
}

void ConvWeights::update(float momentum_decay, float eps) {
  int input_box_count = mask.size();;
  
  momentum_bias = momentum_decay * momentum_bias + eps * deriv_bias;
  bias -= momentum_bias;
  
  float norm2 = bias * bias;
  int count = 1;
  
  assert(isfinite(bias));
  for (int b = 0; b < input_box_count; ++b) {
    momentum_mask[b] = momentum_decay * momentum_mask[b] + eps * deriv_mask[b];
    mask[b] -= momentum_mask[b];
    
    norm2 += mask[b].squaredNorm();
    count += mask[b].rows() * mask[b].cols();
  }
  
  float norm = sqrt(norm2);

  if (norm > 100 * count) {
    norm = 100 * count;
  }
  
  bias /= norm;
  for (int b = 0; b < input_box_count; ++b) {
    mask[b] /= norm;
  }
}


void ConvWeights::setDerivsZero() {
  deriv_bias = 0;
  for (int i = 0; i < deriv_mask.size(); ++i) {
    deriv_mask[i].setZero();
  }
}

