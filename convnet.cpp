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
  
  for (int i = 0; i < layer_count_; ++i) {
    LayerParams::ConnectionType type = params[i].connection_type;
    if (i == 0) {
      assert(type == LayerParams::Initial);
      assert(params[i].box_count == 1);
      layers_[i].initPlain(1, params[i].box_edge);
    } else if (type == LayerParams::Convolution) {
      assert(params[i].mask_edge % 2 == 1);
      int mid = (params[i].mask_edge - 1) / 2;
      assert(params[i].box_edge == params[i - 1].box_edge - 2 * mid);
      layers_[i].initConv(params[i].box_count, params[i].box_edge, params[i - 1].box_count, params[i].mask_edge);
    } else if (type == LayerParams::Pooling) {
      assert(params[i-1].box_edge % 2 == 1); // doing subsampling pooling
      assert((params[i-1].box_edge + 1) / 2 == params[i].box_edge);
      assert(params[i-1].box_count == params[i].box_count);
      layers_[i].initPlain(params[i].box_count, params[i].box_edge);
    } else if (type == LayerParams::SoftMax) {
      assert(params[i].box_count == params[i - 1].box_count);
      assert(params[i].box_edge == params[i - 1].box_edge);
      layers_[i].initPlain(params[i].box_count, params[i].box_edge);
    } else {
      assert(type == LayerParams::Full);
      // For Full connection layer, currently only have support for singleton boxes
      // (because we make use of a single ConvWeights per box to store the weights
      // - to support larger boxes, we'd need a ConvWeights for each node in each box).
      assert(params[layer_count_ - 1].box_edge == 1);
      layers_[i].initConv(params[i].box_count, params[i].box_edge, params[i - 1].box_count, params[i - 1].box_edge);
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

void softMaxBack(Cube const& output_value, Cube const& output_deriv, Cube & input_deriv) {
  // TODO(saxton) implement this
  
  assert(second_edge == 1); // assuming each box is a "singleton" box
  assert(first_edge == 1); // similarly for layer below
  
  
  
  for (int box = 0; box < box_count; ++box) {
    Box & box = layers_[layer].boxes[box_index];
    float value = box.values(0, 0);
    float deriv = box.deriv_values(0, 0);
    for (int box_index_2 = 0; box_index_2 < box_count; ++box_index_2) {
      Box & lower_box = layers_[layer - 1].boxes[box_index_2];
      if (box_index_2 == box_index) {
        lower_box.deriv_values(0, 0) += deriv * value * (1 - value);
      } else {
        float value2 = layers_[layer].boxes[box_index_2].values(0, 0);
        lower_box.deriv_values(0, 0) += - deriv * value * value2;
      }
    }
  }
}

const float leak = 0.0001;

float relu(float x) {
  return x > 0 ? x : leak * x;
}

void convolution(Cube const& input, vector<ConvMask> const& masks, int stride, Cube &output) {
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

void convolutionBack(Cube const& output_value, Cube const& output_deriv, int stride, Cube & input_deriv, vector<ConvMask> mask_deriv) {
  // TODO(saxton) implement this
  
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
	
	mask_deriv[i].bias += deriv_base;
	
        int in_x = j * stride;
        int in_y = k * stride;
	
	// TODO(saxton) check these next two lines
	input_deriv.block(i, in_x, in_y) += deriv_base * mask_deriv[i];
	mask_deriv += deriv_base * input_values.block(0, in_x, in_y);
      }
    }
  }
}

void pooling(Cube const& input, Cube &output) {
  int d0 = input.d0();
  assert(output.d0() == d0);
  int d1 = output.d1();
  int d2 = output.d2();
  assert(input.d1() == 2 * d1 - 1);
  assert(input.d2() == 2 * d2 - 1);
  
  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      for (int k = 0; k < d2; ++k) {
        output(i, j, k) = input(i, 2 * j, 2 * k);
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
    Layer &input = layers_[layer - 1];
    Layer &output = layers_[layer];
    
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
      case LayerParams::Pooling:
        pooling(input.values, output.values);
        continue;
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

void ConvNet::backwardsPass(VectorXf const& target, float learning_rate) {
  // already checked earlier, but do it here too to indicate where it gets used
  assert(params_[layer_count_ - 1].box_edge == 1);
  
  assert(params_[layer_count_ - 1].box_count == target.size());
  
  // Initialize the first set of derivatives, using the error function
  // E = (1/2) \sum_{i = 1}^{N} (v[i] - target[i])^2
  // so {dE}/{dv[i]} = v[i] - target[i]
  Layer * layer = & layers_[layer_count_ - 1];
  for (int i = 0; i < target.size(); ++i) {
    layer.deriv_values(i) = layer.values(i, 0, 0) - target(i);
  }
    
  for (int layer = layer_count_ - 1; layer >= 1; --layer) {
    Layer &input = layers_[layer - 1];
    Layer &output = layers_[layer];
    
    
    switch (params_[layer].connection_type) {
      case LayerParams::Initial:
        assert(false); // should only be initial layer for layer = 0
        continue;
      case LayerParams::Pooling:
        assert(false); // not currently implemented
        continue;
      case LayerParams::SoftMax:
        softMaxBack(output.values, output.deriv_values, input.deriv_values, input.deriv_masks);
        continue;
      case LayerParams::Convolution:
        int stride = params_[layer + 1].connection_type == LayerParams::Pooling ? 2 : 1;
        convolutionBack(input.values, input.masks, stride, output.values);
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


void Layer::initPlain(int box_count, int box_edge) {
  boxes.resize(box_count);
  for (int i = 0; i < box_count; ++i) {
    boxes[i].values.resize(box_edge, box_edge);
    boxes[i].deriv_values.resize(box_edge, box_edge);
  }
}

void Layer::initConv(int box_count, int box_edge, int input_box_count, int mask_edge) {
  initPlain(box_count, box_edge);
  for (int i = 0; i < box_count; ++i) {
    boxes[i].weights.initRandom(input_box_count, mask_edge);
  }
}

void Layer::setDerivsZero() {
  for (int i = 0; i < boxes.size(); ++i) {
    boxes[i].deriv_values.setZero();
    boxes[i].weights.setDerivsZero();
  }
}

void ConvWeights::initRandom(int input_box_count, int mask_edge) {
  mask.resize(input_box_count);
  deriv_mask.resize(input_box_count);
  momentum_mask.resize(input_box_count);
  for (int i = 0; i < input_box_count; ++i) {
    mask[i] = 0.1 * MatrixXf::Random(mask_edge, mask_edge);
    deriv_mask[i] = MatrixXf::Zero(mask_edge, mask_edge);
    momentum_mask[i] = MatrixXf::Zero(mask_edge, mask_edge);
  }
  bias = 0;
  deriv_bias = 0;
}


void ConvWeights::update(float momentum_decay, float eps) {
  int input_box_count = mask.size();
  
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
