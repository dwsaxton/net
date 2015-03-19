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
  
  for (int i = layer_count_ - 1; i >= 0; --i) {
    if (i < layer_count_ - 1 && layers_[i + 1].kernels.size() > 0) {
      int upper_kernel_direction = layers_[i + 1].kernels[0].cube.stackCoordinate();
      layers_[i].value = Cube(params[i].features, params[i].edge, params[i].edge, upper_kernel_direction);
    } else {
      layers_[i].value = Cube(params[i].features, params[i].edge, params[i].edge);
    }
    layers_[i].value_deriv = layers_[i].value;
    
    LayerParams::ConnectionType type = params[i].connection_type;
    
    if (i > 0 && type == LayerParams::Convolution) {
      int features = params[i].features;
      int input_features = params[i - 1].features;
      layers_[i].kernels.resize(features);
      for (int j = 0; j < features; ++j) {
        layers_[i].kernels[j] = Kernel(input_features, params[i].kernel, params[i].kernel);
        layers_[i].kernels[j].setZero();
      }
      layers_[i].kernels_deriv = layers_[i].kernels;
      layers_[i].kernels_momentum = layers_[i].kernels;
    }
    
    layers_[i].randomizeKernels();
    
    // And do some sanity checking
    if (i == 0) {
      assert(type == LayerParams::Initial);
      assert(params[i].features == 1);
    } else if (type == LayerParams::Convolution) {
      assert(params[i].kernel % 2 == 1);
//       int mid = (params[i].kernel - 1) / 2;
//       assert(params[i].edge == params[i - 1].edge - 2 * mid);
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

const float leak = 0.0001;

float relu(float x) {
  return x > 0 ? x : leak * x;
}

void convolution(Cube const& input, vector<Kernel> const& kernels, int stride, Cube &output) {
  int output_features = kernels.size();
  assert(output.d0() == output_features);
  int edge = kernels[0].cube.d1();
  assert(edge == kernels[0].cube.d2()); // only support square kernels
  assert(edge % 2 == 1); // only support odd-sized kernels
  int mid = (edge - 1) / 2;
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
        output(i, j, k) = relu(sum);
      }
    }
  }
}

void convolutionBack(Cube const& output_value, Cube const& output_deriv, int stride, Cube const& input_value, Cube & input_deriv, vector<Kernel const& kernels, vector<Kernel> & kernels_deriv) {
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
        if (output_value(i, j, k) < 0) {
          deriv_base *= leak;
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

void ConvNet::forwardPass(MatrixXf const& input_data) {
  int range_i = params_[0].edge - input_data.rows();
  int range_j = params_[0].edge - input_data.cols();
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
  
  for (int layer = 1; layer < layer_count_; ++layer) {
    Layer const& input = layers_[layer - 1];
    Layer & output = layers_[layer];
    
    switch (params_[layer].connection_type) {
      case LayerParams::Initial:
        // already handled
        continue;
      case LayerParams::SoftMax:
        softMax(input.value, output.value);
        continue;
      case LayerParams::Convolution:
        int stride = params_[layer].stride;
        convolution(input.value, output.kernels, stride, output.value);
        continue;
    }
  }
}

void ConvNet::backwardsPass(VectorXf const& target, float learning_rate) {
  // already checked earlier, but do it here too to indicate where it gets used
  assert(params_[layer_count_ - 1].edge == 1);
  
  assert(params_[layer_count_ - 1].features == target.size());
  
  // Initialize the first set of derivatives, using the error function
  // E = (1/2) \sum_{i = 1}^{N} (v[i] - target[i])^2
  // so {dE}/{dv[i]} = v[i] - target[i]
  
  Layer & top_layer = layers_[layer_count_ - 1];
  for (int i = 0; i < target.size(); ++i) {
    top_layer.value_deriv(i, 0, 0) = top_layer.value(i, 0, 0) - target(i);
  }
    
  for (int layer = layer_count_ - 1; layer >= 1; --layer) {
    Layer &input = layers_[layer - 1];
    Layer &output = layers_[layer];
    
    switch (params_[layer].connection_type) {
      case LayerParams::Initial:
        assert(false); // should only be initial layer for layer = 0
        continue;
      case LayerParams::SoftMax:
        softMaxBack(output.value, output.value_deriv, input.value_deriv);
        continue;
      case LayerParams::Convolution:
        int stride = params_[layer].stride;
        convolutionBack(output.value, output.value_deriv, stride, input.value, input.value_deriv, output.kernels, output.kernels_deriv);
        continue;
    }
  }
  
  for (int layer = 1; layer < layer_count_; ++layer) {
    if (params_[layer].connection_type != LayerParams::SoftMax) {
      layers_[layer].update(MOMENTUM, learning_rate);
    }
  }
}

VectorXf ConvNet::getOutput() const {
  MatrixXf layer = layers_[layer_count_ - 1].value.layer(0);
  if (layer.cols() == 1) {
    return layer.block(0, 0, layer.rows(), 1);
  } else {
    assert(layer.rows() == 1);
    return layer.block(0, 0, 1, layer.cols());
  }
}


void Layer::randomizeKernels() {
  for (int i = 0; i < kernels.size(); ++i) {
    kernels[i].cube.setRandom();
  }
}

void Layer::update(float momentum_decay, float eps) {
  float norm2 = 0;
  int count = 0;
  for (int i = 0; i < kernels.size(); ++i) {
    kernels_momentum[i].scaleAndAddScaled(momentum_decay, eps, kernels_deriv[i]);
    kernels[i] += kernels_momentum[i];
    norm2 += kernels[i].bias * kernels[i].bias + kernels[i].cube.squaredNorm();
    count += 1 + kernels[i].cube.d0() * kernels[i].cube.d1() * kernels[i].cube.d2();
  }

  // TODO this norm division is weird. Makes the kernels very small?
  
  float norm = sqrt(norm2 / count / 4);

  if (norm < 1) {
    norm = 1;
  }
  
  for (int i = 0; i < kernels.size(); ++i) {
    kernels[i] /= norm;
  }
}


void Kernel::scaleAndAddScaled(float scale, float eps, Kernel const& other) {
  assert(cube.height() == other.cube.height());
  bias = scale * bias + eps * other.bias;
  for (int i = 0; i < cube.height(); ++i) {
    cube.layer(i) *= scale;
    cube.layer(i) += eps * other.cube.layer(i);
  }
}

void Kernel::operator+=(Kernel const& other) {
  bias += other.bias;
  cube += other.cube;
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
  Cube kernel(2, 1, 1);
  kernel(0, 0, 0) = 5;
  kernel(1, 0, 0) = 7;
  
  Cube cube(2, 3, 3, kernel.stackCoordinate());
  
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
  
  convolution(cube, kernels, stride, output);
  
  assert(output(0, 0, 0) == 0);
  assert(output(0, 0, 1) == leak * -7 * 5);
  assert(output(0, 1, 0) == 19 * 5);
  assert(output(0, 1, 1) == leak * 25);
}

void TestConvolutionBack() {
//   Cube output_value(1, 2, 2);
//   Cube output_deriv(1, 2, 2);
//   int stride = 1;
//   Cube input_value(1, 2, 2);
//   Cube input_deriv(1, 2, 2);
//   Cube kernel(1, 1, 1);
//   kernel(0, 0, 0) = 1;
//   vector<Kernel> kernels = {kernel};
//   
//   Cube kernel_deriv(1, 1, 1);
//   vector<Kernel> kernels_deriv = {kernel_deriv};
//   
//   output_value(0, 0, 0) = 3;
//   output_value(0, 0, 1) = 7;
//   output_value(0, 1, 0) = -5;
//   output_value(0, 1, 1) = 2;
//   output_deriv(0, 0, 0) = 10;
//   output_deriv(0, 0, 1) = 20;
//   output_deriv(0, 1, 0) = -40;
//   output_deriv(0, 1, 1) = -5;
//   
//   input_value(0, 0, 0) = 0;
//   input_value(0, 0, 1) = 1;
//   input_value(0, 1, 0) = 2;
//   input_value(0, 1, 1) = 3;
//   
//   convolutionBack(output_value, output_deriv, stride, input_value, input_deriv, kernels, kernels_deriv);
}

void TestConvNet() {
  TestSoftMax();
  TestConvolution();
  TestConvolutionBack();
}
