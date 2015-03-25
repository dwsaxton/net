#ifndef CONVNET_H
#define CONVNET_H

#include "cube.h"

class Box;
class Layer;
class LayerParams;

void TestConvNet();
void setLeak(float leak);

class ConvNet {
public:
  ConvNet();
  ConvNet(vector<LayerParams> const& params, float weight_decay);
  
  void setInput(MatrixXf const& input);
  void forwardPass();
  void setTarget(MatrixXf const& target);
  void backwardsPass(float learning_rate);
  VectorXf getOutput() const;
  VectorXf getOutput2() const;
  
  vector<Layer> layers_;
  
private:
  void rescale();
  float weight_decay;
  MatrixXf target_;
};

class LayerParams {
public:
  enum ConnectionType {
    Initial,
    Convolution,
    Scale,
    SoftMax
  };
  
  enum NeuronType {
    ReLU,
    Sigmoid,
  };
  
  LayerParams() { edge = features = kernel = stride = 1; connection_type = Initial; neuron_type = ReLU; }
  
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
  // For use with Convolution, the neuron function to use
  NeuronType neuron_type;
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
  void addScaled(float eps, Kernel const& other);
  void scaleAndDivideByCwiseSqrt(float scale, Kernel const& other);
  void addCwiseSquare(Kernel const& other);
  void operator+=(Kernel const& other);
  void operator-=(Kernel const& other);
  void operator/=(float v);
};

class Layer {
public:
  Layer();
  Layer(LayerParams const& params, int input_features, int stack_coordinate = -1);
  int features() const { return kernels.size(); }
  void update(float momentum_decay, float eps, float weight_decay);
  
  Cube value;
  Cube value_deriv;
  vector<Kernel> kernels;
  vector<Kernel> kernels_deriv;
  vector<Kernel> kernels_momentum;
  LayerParams params;
};

#endif
