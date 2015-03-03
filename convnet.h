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
    Full
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
};

class ConvWeights {
public:
  ConvWeights() { bias = deriv_bias = momentum_bias = 0; }
  
  void initRandom(int input_box_count, int mask_edge);
  
  vector<MatrixXf> mask;
  float bias;
  
  vector<MatrixXf> deriv_mask;
  float deriv_bias;
  
  vector<MatrixXf> momentum_mask;
  float momentum_bias;
  
  void setDerivsZero();
  
  // returns sigmoid(<mask, input-with-top-left-at-x-and-y> + bias).
  float sigmoidOfConv(Layer const& input, const int x, const int y) const;
  void doSigmoidOfConvDeriv(Layer& input, const int x, const int y, float value, float deriv_value);
  void update(float momentum_decay, float eps);
};

class Layer {
public:
  int boxCount() const { return boxes.size(); }
  
  void setDerivsZero();
  
  vector<Box> boxes;
  
  // Initializes the layer to have box_count boxes, where each box has edge size box_edge
  void initPlain(int box_count, int box_edge);
  // Does initPlain, and in addition initializes the convolution masks randomly using conv_edge size
  void initConv(int box_count, int box_edge, int input_box_count, int mask_edge);
};

class Box {
public:
  int edge() const { return values.rows(); }
  
  // The values of the nodes in this box
  MatrixXf values;
  // In the case that this layer has a Convolution connection with the previous layer, then the weights used.
  ConvWeights weights;
  // For back propagation
  MatrixXf deriv_values;
};

#endif
