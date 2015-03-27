#include "convnet.h"
#include "cube.h"

#include <cassert>
#include <iostream>
using namespace std;

const float MOMENTUM = 0.9;

ConvNet::ConvNet() {
}

ConvNet::ConvNet(vector<LayerParams> const& params, float weight_decay) {
  this->weight_decay = weight_decay;
  assert(params.size() >= 1);
  layers_.resize(params.size());
  
  for (int i = layers_.size() - 1; i >= 0; --i) {
    int input_features = i > 0 ? params[i - 1].features : 0;
    if (i < layers_.size() - 1 && layers_[i + 1].kernels.size() > 0) {
      int upper_kernel_direction = layers_[i + 1].kernels[0].cube.stackCoordinate();
      layers_[i] = Layer(params[i], input_features, upper_kernel_direction);
    } else {
      layers_[i] = Layer(params[i], input_features);
    }
    
    // And do some sanity checking
    LayerParams::ConnectionType type = params[i].connection_type;
    if (i == 0) {
      assert(type == LayerParams::Initial);
      assert(params[i].features == 1);
    } else if (type == LayerParams::Convolution) {
//       assert(params[i].kernel % 2 == 1);
//       int mid = (params[i].kernel - 1) / 2;
//       assert(params[i].edge == params[i - 1].edge - 2 * mid);
    } else if (type == LayerParams::Scale) {
      assert(params[i].edge == params[i - 1].edge);
      assert(params[i].features == params[i - 1].features);
    } else {
      assert(type == LayerParams::SoftMax);
      assert(params[i].features == params[i - 1].features);
      assert(params[i].edge == params[i - 1].edge);
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
  MatrixXf const& output_value = output_cube_value.layer(0);
  MatrixXf const& output_deriv = output_cube_deriv.layer(0);
  MatrixXf & input_deriv = input_cube_deriv.layer(0);
  input_deriv = -output_value.cwiseProduct(output_deriv).sum() * output_value + output_value.cwiseProduct(output_deriv);
}

float getScaleFactor(Cube const& values) {
  float mx = values.maxCoeff();
  float min = values.minCoeff();
  float factor = max(mx, -min);
  if (factor < 1) {
    factor = 1;
  }
  return factor;
}

void scale(float multiplier, Cube const& input_cube, Cube &output_cube) {
  output_cube = input_cube;
  output_cube /= getScaleFactor(input_cube) / multiplier;
}

void scaleBack(float multiplier, Cube const& output_cube_deriv, Cube const& input_cube_value, Cube & input_cube_deriv) {
  float factor = getScaleFactor(input_cube_value) / multiplier;
  input_cube_deriv = output_cube_deriv;
  input_cube_deriv /= factor;
  
  bool found = false;
  int max_i = -1;
  int max_j = -1;
  int max_k = -1;
  
  for (int i = 0; i < input_cube_value.d0() && !found; ++i) {
    for (int j = 0; j < input_cube_value.d1() && !found; ++j) {
      for (int k = 0; k < input_cube_value.d2(); ++k) {
        if (abs(input_cube_value(i, j, k)) / multiplier == factor) {
          max_i = i;
          max_j = j;
          max_k = k;
          found = true;
          break;
        }
      }
    }
  }
  
  if (!found) {
    // the scale factor was probably capped, so couldn't be found;
    // therefore since there's no element that bites, don't add this part
    // of the derivative
//     cout << "@Not found!" << endl;
    return;
  }
  
  float value = input_cube_value(max_i, max_j, max_j);
  
  // For now, only done single-height implementation
  assert(input_cube_value.height() == 1); //
  input_cube_deriv(max_i, max_j, max_k) -=
      output_cube_deriv.layer(0).cwiseProduct(input_cube_value.layer(0)).sum() / value / factor;
}

float leak = 0.01;
void setLeak(float l) {
  leak = l;
}

float relu(float x) {
  return x > 0 ? x : leak * x;
}

float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

void convolution(LayerParams::NeuronType neuron_type, Cube const& input, vector<Kernel> const& kernels, int stride, Cube &output) {
  int output_features = kernels.size();
  assert(output.d0() == output_features);
  int edge = kernels[0].cube.d1();
  assert(edge == kernels[0].cube.d2()); // only support square kernels
  assert(kernels[0].cube.d0() == input.d0());
  
  assert(input.d1() == input.d2()); // only support square inputs
  assert(output.d1() == output.d2()); // only support square outputs
  
  assert((input.d1() - 1) % stride == 0);
  assert(output.d1() == (input.d1() - edge) / stride + 1);
  
  for (int i = 0; i < output_features; ++i) {
    for (int j = 0; j < output.d1(); ++j) {
      for (int k = 0; k < output.d2(); ++k) {
        int in_x = j * stride;
        int in_y = k * stride;
        float sum = kernels[i].bias + input.computeKernel(kernels[i].cube, in_x, in_y);
        output(i, j, k) = neuron_type == LayerParams::ReLU ? relu(sum) : sigmoid(sum);
      }
    }
  }
}

void convolutionBack(
    LayerParams::NeuronType neuron_type,
    Cube const& output_value,
    Cube const& output_deriv,
    int stride,
    Cube const& input_value,
    Cube & input_deriv,
    vector<Kernel> const& kernels,
    vector<Kernel> & kernels_deriv) {
  int output_features = kernels_deriv.size();
  assert(output_features == output_value.d0());
  
  input_deriv.setZero();
  for (int i = 0; i < kernels_deriv.size(); ++i) {
    kernels_deriv[i].setZero();
  }
  
  for (int i = 0; i < output_features; ++i) {
    for (int j = 0; j < output_value.d1(); ++j) {
      for (int k = 0; k < output_value.d2(); ++k) {
        float deriv_base = output_deriv(i, j, k);
        float v = output_value(i, j, k);
        if (neuron_type == LayerParams::ReLU) {
          if (v < 0) {
            deriv_base *= leak;
          }
        } else {
          deriv_base *= v * (1 - v);
        }
    
        int in_x = j * stride;
        int in_y = k * stride;
    
        input_deriv.addScaledKernel(deriv_base, kernels[i].cube, in_x, in_y);
        kernels_deriv[i].bias += deriv_base;
        kernels_deriv[i].cube.addScaledSubcube(deriv_base, input_value, in_x, in_y); // += deriv_base 
      }
    }
  }
}

void ConvNet::setInput(MatrixXf const& input_data) {
  int range_i = layers_[0].params.edge - input_data.rows();
  int range_j = layers_[0].params.edge - input_data.cols();
  assert(range_i >= 0);
  assert(range_j >= 0);
  int offset_i = range_i == 0 ? 0 : rand() % range_i;
  int offset_j = range_j == 0 ? 0 : rand() % range_j;
  if (range_i == 0 && range_j == 0) {
    layers_[0].value.layer(0) = input_data;
  } else {
    layers_[0].value.layer(0).setZero();
    layers_[0].value.layer(0).block(offset_i, offset_j, input_data.rows(), input_data.cols()) = input_data;
  }
}

void ConvNet::setInput(VectorXf const& input) {
  assert(layers_[0].value.d0() == input.size());
  assert(layers_[0].value.d1() == 1);
  assert(layers_[0].value.d2() == 1);
  
  for (int i = 0; i < input.size(); ++i) {
    layers_[0].value(i, 0, 0) = input(i);
  }
}

void ConvNet::forwardPass() {
  for (int layer = 1; layer < layers_.size(); ++layer) {
    Layer const& input = layers_[layer - 1];
    Layer & output = layers_[layer];
    
    switch (output.params.connection_type) {
      case LayerParams::Initial:
        // already handled
        continue;
      case LayerParams::Scale:
        scale(output.params.scale, input.value, output.value);
        continue;
      case LayerParams::SoftMax:
        softMax(input.value, output.value);
        continue;
      case LayerParams::Convolution:
        int stride = output.params.stride;
        convolution(output.params.neuron_type, input.value, output.kernels, stride, output.value);
        continue;
    }
  }
}

void ConvNet::setTarget(int target) {
  target_int_ = target;
}

void ConvNet::setTarget(MatrixXf const& target) {
  target_matrix_ = target;
  target_int_ = -1;
}

void ConvNet::backwardsPass(float learning_rate) {
  if (target_int_ >= 0) {
    // already checked earlier, but do it here too to indicate where it gets used
    assert(layers_[layers_.size() - 1].params.edge == 1);
    
    int top_features = layers_[layers_.size() - 1].params.features;
    
    // Initialize the first set of derivatives, using the error function
    // E = (1/2) \sum_{i = 1}^{N} (v[i] - target[i])^2
    // so {dE}/{dv[i]} = v[i] - target[i]
    
    Layer & top_layer = layers_[layers_.size() - 1];
    for (int i = 0; i < top_features; ++i) {
      float v = top_layer.value(i, 0, 0);
      float t = i == target_int_ ? 1 : 0;
      float weight = (i == target_int_) ? top_features - 1 : 1;
      top_layer.value_deriv(i, 0, 0) = weight * (v - t);
    }
  } else {
    int rows = target_matrix_.rows();
    int cols = target_matrix_.cols();
    Layer & top_layer = layers_[layers_.size() - 1];
    // Just assumptions we've used for implementing this function
    assert(top_layer.value.d0() == rows * cols);
    assert(top_layer.value.d1() == 1);
    assert(top_layer.value.d2() == 1);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int k = i * cols + j;
        float v = top_layer.value(k, 0, 0);
        float t = target_matrix_(i, j);
        top_layer.value_deriv(k, 0, 0) = v - t;
      }
    }
  }
  
  for (int layer = layers_.size() - 1; layer >= 1; --layer) {
    Layer &input = layers_[layer - 1];
    Layer &output = layers_[layer];
    
    switch (output.params.connection_type) {
      case LayerParams::Initial:
        assert(false); // should only be initial layer for layer = 0
        continue;
      case LayerParams::Scale:
        scaleBack(output.params.scale, output.value_deriv, input.value, input.value_deriv);
        continue;
      case LayerParams::SoftMax:
        softMaxBack(output.value, output.value_deriv, input.value_deriv);
        continue;
      case LayerParams::Convolution:
        int stride = output.params.stride;
        convolutionBack(output.params.neuron_type, output.value, output.value_deriv, stride, input.value, input.value_deriv, output.kernels, output.kernels_deriv);
        continue;
    }
  }
  
  float max_abs = 1;
  for (int layer = 1; layer < layers_.size(); ++layer) {
    if (layers_[layer].params.connection_type == LayerParams::Convolution) {
      for (int i = 0; i < layers_[layer].kernels_deriv.size(); ++i) {
        max_abs = max(max_abs, layers_[layer].kernels_deriv[i].cube.maxCoeff());
        max_abs = max(max_abs, -layers_[layer].kernels_deriv[i].cube.minCoeff());
      }
    }
  }
  
  for (int layer = 1; layer < layers_.size(); ++layer) {
    if (layers_[layer].params.connection_type == LayerParams::Convolution) {
      layers_[layer].update(MOMENTUM, learning_rate / max_abs, weight_decay);
    }
  }
}

VectorXf ConvNet::getOutputPrior(int from_top) const {
  MatrixXf layer = layers_[layers_.size() - 1 - from_top].value.layer(0);
  if (layer.cols() == 1) {
    return layer.block(0, 0, layer.rows(), 1);
  } else {
    assert(layer.rows() == 1);
    return layer.block(0, 0, 1, layer.cols());
  }
}

VectorXf ConvNet::getOutput() const {
  return getOutputPrior(0);
}

VectorXf ConvNet::getOutput2() const {
  return getOutputPrior(1);
}

Layer::Layer() {
}

Layer::Layer(LayerParams const& params, int input_features, int stack_coordinate) {
  this->params = params;
  if (stack_coordinate >= 0) {
    value = Cube(params.features, params.edge, params.edge, stack_coordinate);
  } else {
    value = Cube(params.features, params.edge, params.edge);
  }
  value_deriv = value;
  
  LayerParams::ConnectionType type = params.connection_type;
  
  if (type == LayerParams::Convolution) {
    int features = params.features;
    kernels.resize(features);
    for (int j = 0; j < features; ++j) {
      kernels[j] = Kernel(input_features, params.kernel, params.kernel);
      kernels[j].setZero();
    }
    kernels_deriv = kernels;
    kernels_momentum = kernels;
    
    for (int j = 0; j < features; ++j) {
      kernels[j].cube.setRandom();
    }
  }
}

void Layer::update(float momentum_decay, float eps, float weight_decay) {
  for (int i = 0; i < kernels.size(); ++i) {
//     kernels_deriv[i].scaleAndDivideByCwiseSqrt(eps, kernels_adagrad[i]);
//     kernels_adagrad[i].addCwiseSquare(kernels_deriv[i]);
//     kernels[i] -= kernels_deriv[i];
    
    kernels_momentum[i].scaleAndAddScaled(momentum_decay, eps, kernels_deriv[i]);
    kernels[i] -= kernels_momentum[i];
    kernels[i].addScaled(eps * weight_decay, kernels[i]);
    
//     kernels[i].addScaled(-eps, kernels_deriv[i]);

//     if (rand() % 400000 == 0) {
//       cout << "n2: " << kernels[i].cube.squaredNorm() << endl;
//     }
  }
  
//   for (int i = 0; i < kernels.size(); ++i) {
//     kernels[i] /= norm;
//   }
}


void Kernel::scaleAndAddScaled(float scale, float eps, Kernel const& other) {
  assert(cube.height() == other.cube.height());
  bias = scale * bias + eps * other.bias;
  for (int i = 0; i < cube.height(); ++i) {
    cube.layer(i) *= scale;
    cube.layer(i) += eps * other.cube.layer(i);
  }
}

void Kernel::addScaled(float eps, Kernel const& other) {
  assert(cube.height() == other.cube.height());
  bias += eps * other.bias;
  for (int i = 0; i < cube.height(); ++i) {
    cube.layer(i) += eps * other.cube.layer(i);
  }
}

void Kernel::scaleAndDivideByCwiseSqrt(float scale, Kernel const& other) {
  bias = (scale * bias) / sqrt(other.bias);
  for (int i = 0; i < cube.height(); ++i) {
    cube.layer(i) *= scale;
    cube.layer(i).array() /= other.cube.layer(i).array().sqrt();
  }
}

void Kernel::addCwiseSquare(Kernel const& other) {
  bias += other.bias * other.bias;
  for (int i = 0; i < cube.height(); ++i) {
    cube.layer(i).array() += other.cube.layer(i).array().square();
  }
}

void Kernel::operator+=(Kernel const& other) {
  bias += other.bias;
  cube += other.cube;
}

void Kernel::operator-=(Kernel const& other) {
  bias -= other.bias;
  cube -= other.cube;
}

void Kernel::operator/=(float v) {
  bias /= v;
  cube /= v;
}


void TestSoftMax() {
//   void softMax(Cube const& input, Cube &output) {
  
//   void softMaxBack(Cube const& output_cube_value, Cube const& output_cube_deriv, Cube & input_cube_deriv) {
}

void TestConvolution() {
  Kernel kernel;
  kernel.cube = Cube(2, 1, 1);
  kernel.cube(0, 0, 0) = 5;
  kernel.cube(1, 0, 0) = 7;
  kernel.bias = 1;
  
  Cube cube(2, 3, 3, kernel.cube.stackCoordinate());
  
  cube(0, 0, 0) = 0;
  cube(0, 0, 1) = 1;
  cube(0, 1, 0) = 2;
  cube(0, 1, 1) = 3;
  cube(0, 2, 2) = -5;
  cube(0, 0, 2) = -7;
  cube(0, 2, 0) = 19;
  
  cube(1, 0, 0) = 0;
  cube(1, 0, 1) = 1;
  cube(1, 1, 0) = 2;
  cube(1, 1, 1) = 3;
  
  vector<Kernel> kernels = {kernel};
  int stride = 2;
  Cube output(1, 2, 2);
  
  convolution(LayerParams::ReLU, cube, kernels, stride, output);
  
  assert(output(0, 0, 0) == 1);
  assert(output(0, 0, 1) == leak * (-7 * 5 + 1));
  assert(output(0, 1, 0) == 19 * 5 + 1);
  assert(output(0, 1, 1) == leak * (-25 + 1));
}

void TestConvolutionBack() {
  Cube output_value(1, 2, 2);
  Cube output_deriv(1, 2, 2);
  int stride = 1;
  Cube input_value(1, 2, 2);
  Cube input_deriv(1, 2, 2);
  Kernel kernel;
  kernel.cube = Cube(1, 1, 1);
  kernel.cube(0, 0, 0) = 2;
  kernel.bias = 3;
  vector<Kernel> kernels = {kernel};
  
  vector<Kernel> kernels_deriv(1);
  Kernel &kernel_deriv = kernels_deriv[0];
  kernel_deriv.cube = Cube(1, 1, 1);
  
  output_value(0, 0, 0) = 3;
  output_value(0, 0, 1) = 7;
  output_value(0, 1, 0) = -5;
  output_value(0, 1, 1) = 2;
  output_deriv(0, 0, 0) = 10;
  output_deriv(0, 0, 1) = 20;
  output_deriv(0, 1, 0) = -40;
  output_deriv(0, 1, 1) = -5;
  
  input_value(0, 0, 0) = 0;
  input_value(0, 0, 1) = 1;
  input_value(0, 1, 0) = 2;
  input_value(0, 1, 1) = 3;
  
  convolutionBack(LayerParams::ReLU, output_value, output_deriv, stride, input_value, input_deriv, kernels, kernels_deriv);
  
  assert(abs(kernel_deriv.cube(0,0,0) - (20 - 15 - 2 * 40 * leak)) < 0.00001);
  assert(abs(kernel_deriv.bias - (10 + 20 - 40 * leak - 5)) < 0.00001);
  assert(input_deriv(0, 0, 0) == 20);
  assert(abs(input_deriv(0, 1, 0) + leak * 80) < 0.0001);
}

void TestConvNet() {
  TestSoftMax();
  TestConvolution();
  TestConvolutionBack();
}
