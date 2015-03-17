#ifndef CONVNET_H
#define CONVNET_H

#include <Eigen/Geometry>
using namespace Eigen;

#include <vector>
using namespace std;

class Box;
class Layer;
class LayerParams;

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
    Pooling,
    SoftMax
  };
  
  LayerParams() { box_edge = box_count = mask_edge = 0; connection_type = Initial; }
  
  // Layer is composed of a collection of n x n nodes; this is the value of n
  int box_edge;
  // Number of n x n squares in this layer
  int box_count;
  // Connection with the previous layer
  ConnectionType connection_type;
  // For Convolution or Full connection with the previous layer, the value of b in the b x b template
  int mask_edge;
  // For convolution, the spacing between mask samplings
  int stride;
};

class ConvMask {
public:
  Cube kernel;
  float bias;
  
  void setZero() {
    bias = 0;
    kernel.setZero();
  }
}

class Layer {
public:
  int boxCount() const { return conv_weights.size(); }
  
  Cube values;
  Cube deriv_values;
  vector<ConvMask> masks;
  vector<ConvMask> deriv_masks;
  
  // Initializes the layer to have box_count boxes, where each box has edge size box_edge
  void initPlain(int box_count, int box_edge);
  // Does initPlain, and in addition initializes the convolution masks randomly using conv_edge size
  void initConv(int box_count, int box_edge, int input_box_count, int mask_edge);
};

#endif
