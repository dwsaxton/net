#include "convnet.h"

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

// float sigmoid(float x) {
//   return 1 / (1 + exp(-x));
// }

void ConvNet::forwardPass(MatrixXf const& input) {
  int range_i = params_[0].box_edge - input.rows();
  int range_j = params_[0].box_edge - input.cols();
  assert(range_i >= 0);
  assert(range_j >= 0);
  int offset_i = range_i == 0 ? 0 : rand() % range_i;
  int offset_j = range_j == 0 ? 0 : rand() % range_j;
  if (range_i == 0 && range_j == 0) {
    layers_[0].boxes[0].values = input;
  } else {
    layers_[0].boxes[0].values.setZero();
    layers_[0].boxes[0].values.block(offset_i, offset_j, input.rows(), input.cols()) = input;
  }
  
  for (int layer = 1; layer < layer_count_; ++layer) {
    int first_edge = params_[layer - 1].box_edge;
    int second_edge = params_[layer].box_edge;
    int box_count = params_[layer].box_count;
    
    if (params_[layer].connection_type == LayerParams::SoftMax) {
      float sum = 0;
      assert(first_edge == second_edge);
      assert(second_edge == 1); // required assumption of current implementation
      
      // use max_value to divide the top and bottom of the fraction in SoftMax by exp(max_value),
      // to avoid float-overflow
      float max_value = -HUGE_VAL;
      for (int box_index = 0; box_index < box_count; ++box_index) {
        max_value = max(max_value, layers_[layer - 1].boxes[box_index].values(0, 0));
      }
      
      for (int box_index = 0; box_index < box_count; ++box_index) {
        Box & box = layers_[layer].boxes[box_index];
        Box & prev_box = layers_[layer - 1].boxes[box_index];
        float value = exp(prev_box.values(0, 0) - max_value);
        if (!isfinite(value)) {
          cout << " prev_box.values(i, j)=" << prev_box.values(0, 0) << endl;
        }
        sum += value;
        box.values(0, 0) = value;
      }
      for (int box_index = 0; box_index < box_count; ++box_index) {
        Box & box = layers_[layer].boxes[box_index];
        box.values /= sum;
        if (!isfinite(box.values.norm())) {
          cout << "sum=" << sum << endl;
        }
      }
      continue;
    }
    
    for (int box_index = 0; box_index < box_count; ++box_index) {
      Box & box = layers_[layer].boxes[box_index];
      
      if (params_[layer].connection_type == LayerParams::Full) { 
        assert(second_edge == 1); // assuming each box is a "singleton" box
        box.values(0, 0) =
            box.weights.sigmoidOfConv(layers_[layer - 1], 0, 0);
        continue;
      }
      
      if (params_[layer].connection_type == LayerParams::Pooling) {
        for (int i = 0; i < second_edge; ++i) {
          for (int j = 0; j < second_edge; ++j) {
            Box & prev_box = layers_[layer - 1].boxes[box_index];
            box.values(i, j) = prev_box.values(2 * i, 2 * j);
          }
        }
        continue;
      }
      
      assert(params_[layer].connection_type == LayerParams::Convolution);
      int mask_edge = params_[layer].mask_edge;
      assert(mask_edge % 2 == 1);
      int mid = (mask_edge - 1) / 2;
      for (int i = 0; i < second_edge; ++i) {
        for (int j = 0; j < second_edge; ++j) {
          if (params_[layer + 1].connection_type == LayerParams::Pooling && (i % 2 == 1 || j % 2 == 1)) {
            continue;
          }
          
          box.values(i, j) = 
              box.weights.sigmoidOfConv(layers_[layer - 1], i, j);
        }
      }
    }
  }
}

VectorXf ConvNet::getOutput() const {
  int count = params_[layer_count_ - 1].box_count;
  VectorXf output(count);
  for (int i = 0; i < count; ++i) {
    output[i] = layers_[layer_count_ - 1].boxes[i].values(0, 0);
  }
  return output;
}

VectorXf ConvNet::get2ndOutput() const {
  int count = params_[layer_count_ - 2].box_count;
  VectorXf output(count);
  for (int i = 0; i < count; ++i) {
    output[i] = layers_[layer_count_ - 2].boxes[i].values(0, 0);
  }
  return output;
}

void ConvNet::backwardsPass(VectorXf const& target, float learning_rate) {
  // already checked earlier, but do it here too to indicate where it gets used
  assert(params_[layer_count_ - 1].box_edge == 1);
  
  assert(params_[layer_count_ - 1].box_count == target.size());
  
  for (int i = 0; i < layer_count_; ++i) {
    layers_[i].setDerivsZero();
  }
  
  // Initialize the first set of derivatives, using the error function
  // E = (1/2) \sum_{i = 1}^{N} (v[i] - target[i])^2
  // so {dE}/{dv[i]} = v[i] - target[i]
  for (int i = 0; i < target.size(); ++i) {
    Box & box = layers_[layer_count_ - 1].boxes[i];
    box.deriv_values(0, 0) = box.values(0, 0) - target(i);
  }
  
  for (int layer = layer_count_ - 1; layer >= 1; --layer) {
    int first_edge = params_[layer - 1].box_edge;
    int second_edge = params_[layer].box_edge;
    int box_count = params_[layer].box_count;
    
    if (params_[layer].connection_type == LayerParams::SoftMax) {
      assert(second_edge == 1); // assuming each box is a "singleton" box
      assert(first_edge == 1); // similarly for layer below
      
      for (int box_index = 0; box_index < box_count; ++box_index) {
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
      continue;
    }
    
    for (int box_index = 0; box_index < box_count; ++box_index) {
      Box & box = layers_[layer].boxes[box_index];
      if (params_[layer].connection_type == LayerParams::Full) { 
        assert(second_edge == 1); // assuming each box is a "singleton" box
        float value = box.values(0, 0);
        float deriv_value = box.deriv_values(0, 0);
        box.weights.doSigmoidOfConvDeriv(layers_[layer - 1], 0, 0, value, deriv_value);
        continue;
      }
      
      if (params_[layer].connection_type == LayerParams::Pooling) {
        for (int i = 0; i < second_edge; ++i) {
          for (int j = 0; j < second_edge; ++j) {
            Box & prev_box = layers_[layer - 1].boxes[box_index];
            prev_box.deriv_values(2 * i, 2 * j) = box.deriv_values(i, j);
          }
        }
        continue;
      }
      
      assert(params_[layer].connection_type == LayerParams::Convolution);
      int mask_edge = params_[layer].mask_edge;
      assert(mask_edge % 2 == 1);
      int mid = (mask_edge - 1) / 2;
      for (int i = 0; i < second_edge; ++i) {
        for (int j = 0; j < second_edge; ++j) {
          if (params_[layer + 1].connection_type == LayerParams::Pooling && (i % 2 == 1 || j % 2 == 1)) {
            continue;
          }
          float value = box.values(i, j);
          float deriv_value = box.deriv_values(i, j);
          box.weights.doSigmoidOfConvDeriv(layers_[layer - 1], i, j, value, deriv_value);
        }
      }
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

const float leak = 0.0001;

float ConvWeights::sigmoidOfConv(Layer const& input, const int x, const int y) const {
  int input_box_count = input.boxCount();
  assert(input_box_count == mask.size());
  
  int mask_edge = mask[0].rows();
  assert(mask_edge > 0);
  
  float sum = bias;
  for (int b = 0; b < input_box_count; ++b) {
    sum += input.boxes[b].values.block(x, y, mask_edge, mask_edge).cwiseProduct(mask[b]).sum();
  }
  
  assert(isfinite(sum));
  if (sum >= 0) {
    return sum;
  } else {
    return leak * sum;
  }
}

void ConvWeights::doSigmoidOfConvDeriv(Layer& input, const int x, const int y, float value, float deriv_value) {
  int input_box_count = input.boxCount();
  assert(input_box_count == mask.size());
  
  int mask_edge = mask[0].rows();
  assert(mask_edge > 0);
  
  float base = deriv_value;
  if (value < 0) {
    base *= leak;
  }
  
  deriv_bias = base;
  assert(isfinite(base));
  for (int b = 0; b < input_box_count; ++b) {
    deriv_mask[b] += base * input.boxes[b].values.block(x, y, mask_edge, mask_edge);
    input.boxes[b].deriv_values.block(x, y, mask_edge, mask_edge) += base * mask[b];
  }
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
