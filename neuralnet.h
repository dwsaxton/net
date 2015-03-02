#ifndef NEURALNET_H
#define NEURALNET_H

#include <Eigen/Geometry>
using namespace Eigen;

#include <vector>
using namespace std;

class NeuralNet {
public:
  NeuralNet() {}
  NeuralNet(vector<int> const& N);

  void calcDeriv(VectorXd const& target);
  // Initializes the weights at random
  void initRandom();
  // Initializes the connections between layer i and i+1 to random (i starts at 0)
  void initRandom(int i);
  // Calculates the activitations from the first layer a_[0]
  void calcActivations();
  void calcActivationsWithBinaryRestriction();
  void subtractDeriv(double eps);

  vector<MatrixXd> deriv_;
  vector<int> N_;
  vector<MatrixXd> w_;

  void setInput(VectorXd const& input);
  VectorXd output() const { return a_[N_.size() - 1]; }

  // Train the outer most layer using the given input
  void doContrastiveDivergence(double eps);
  // Adds one more layer to the neural net
  void addLayer(int size);

private:
  vector<VectorXd> a_;
};

#endif // NEURALNET_H
