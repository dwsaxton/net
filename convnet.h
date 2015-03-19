#ifndef CONVNET_H
#define CONVNET_H

#include "cube.h"

class Box;
class Layer;
class LayerParams;

void TestConvNet();

class ConvNet {
public:
  ConvNet();
  ConvNet(vector<LayerParams> const& params);
  
  void forwardPass(MatrixXf const& input);
  void backwardsPass(const VectorXf& target, float learning_rate);
  VectorXf getOutput() const;
  
// private:
  void updateWeights();
  
  vector<LayerParams> params_;
  vector<Layer> layers_;
  int layer_count_;
};

class LayerParams {
public:
  enum ConnectionType {
    Initial,
    Convolution,
    SoftMax
  };
  
  LayerParams() { edge = features = kernel = stride = 1; connection_type = Initial; }
  
  // Connection with the previous layer
  ConnectionType connection_type;
  // Layer is composed of a collection of n x n nodes; this is the value of n
  int edge;
  // Number of n x n squares in this layer
  int features;
  // For Convolution or Full connection with the previous layer, the value of b in the b x b template
  int kernel;
  // For convolution, the spacing between mask samplings
  int stride;
};

class Kernel {
public:
  Kernel() { setZero(); }
  Kernel(int d0, int d1, int d2) {
    cube = Cube(d0, d1, d2);
    setZero();
  }
  
  Cube cube;
  float bias;
  
  void setZero() {
    bias = 0;
    cube.setZero();
  }
  
  /**
   * Replaces this with scale * this + eps * other.
   */
  void scaleAndAddScaled(float scale, float eps, Kernel const& other);
  void operator+=(Kernel const& other);
  void operator/=(float v);
};

class Layer {
public:
  int features() const { return kernels.size(); }
  void randomizeKernels();
  void update(float momentum_decay, float eps);
  
  Cube value;
  Cube value_deriv;
  vector<Kernel> kernels;
  vector<Kernel> kernels_deriv;
  vector<Kernel> kernels_momentum;
};

#endif
