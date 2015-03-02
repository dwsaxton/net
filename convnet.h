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
  ConvNet(vector<LayerParams> const& params);
  
  void forwardPass(MatrixXd const& input);
  void backwardsPass(VectorXd const& target);
  VectorXd getOutput() const;
  
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
  
  vector<MatrixXd> mask;
  double bias;
  
  vector<MatrixXd> deriv_mask;
  double deriv_bias;
  
  vector<MatrixXd> momentum_mask;
  double momentum_bias;
  
  void setDerivsZero();
  
  // returns sigmoid(<mask, input-with-top-left-at-x-and-y> + bias).
  double sigmoidOfConv(Layer const& input, const int x, const int y) const;
  void doSigmoidOfConvDeriv(Layer& input, const int x, const int y, double value, double deriv_value);
  void update(double momentum_decay, double eps);
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
  MatrixXd values;
  // In the case that this layer has a Convolution connection with the previous layer, then the weights used.
  ConvWeights weights;
  // For back propagation
  MatrixXd deriv_values;
};

#endif
