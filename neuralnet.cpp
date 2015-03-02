#include "neuralnet.h"

#include <cstdlib>
#include <iostream>
using namespace std;

NeuralNet::NeuralNet(vector<int> const& N) {
  assert(N.size() >= 2);
  N_ = N;
  w_.resize(N.size() - 1);
  for (int i = 0; i < N.size() - 1; ++i) {
    w_[i].resize(N[i] + 1, N[i+1]);
    w_[i].setZero();
  }
  deriv_ = w_;

  a_.resize(N.size());
  for (int i = 0; i < N.size(); ++i) {
    int last = i == N.size() - 1;
    a_[i].resize(last ? N[i] : N[i]+1);
    a_[i].setZero();
    if (!last) {
      a_[i][N[i]] = 1; // the constant bias input
    }
  }
}

void NeuralNet::setInput(VectorXd const& input) {
  int count = N_[0];
  assert(input.size() == count);
  a_[0].block(0, 0, count, 1) = input;
}

void NeuralNet::calcActivations() {
  for (int i = 1; i < N_.size(); ++i) {
    for (int k = 0; k < N_[i]; ++k) {
      double in = a_[i-1].dot(w_[i-1].col(k));
      a_[i][k] = 1 / (1 + exp(-in));
    }
  }
}

int sampleBernoulli(double p) {
  return (1.0 * rand() / RAND_MAX) < p ? 0 : 1;
}

void NeuralNet::calcActivationsWithBinaryRestriction() {
  for (int i = 1; i < N_.size(); ++i) {
    for (int k = 0; k < N_[i]; ++k) {
      double in = a_[i-1].dot(w_[i-1].col(k));
      double p = 1 / (1 + exp(-in));
      a_[i][k] = sampleBernoulli(p);
    }
  }
}


void NeuralNet::calcDeriv(VectorXd const& target) {
  int layers = N_.size();
  vector<VectorXd> derivs(layers - 1);
  
  // First initialize the top layer with the derivatives from the energy function
  derivs[layers - 2].resize(N_[layers - 1]);
  for (int i = 0; i < N_[layers - 1]; ++i) {
    derivs[layers - 2][i] = a_[layers - 1][i] - target[i];
  }

  // Reset the derivatives to zero
  for (int i = 0; i < layers - 1; ++i) {
    deriv_[i].setZero();
  }

  double weight_penalty = 1e-4;

  // Now do back propagation
  for (int i = layers - 2; i >= 0; --i) {
    for (int j = 0; j < N_[i] + 1; ++j) {
      for (int k = 0; k < N_[i + 1]; ++k) {
        double x = a_[i][j];
        double y = a_[i+1][k];
        deriv_[i](j, k) = derivs[i][k] * x * y * (1 - y) + weight_penalty * w_[i](j, k);
      }
    }

    if (i > 0) {
      derivs[i-1].resize(N_[i]);
      derivs[i-1].setZero();
      for (int j = 0; j < N_[i]; ++j) {
        for (int k = 0; k < N_[i + 1]; ++k) {
          double z = a_[i + 1][k];
          derivs[i-1][j] += derivs[i][k] * w_[i](j, k) * z * (1 - z);
        }
      }
    }
  }
}

void NeuralNet::initRandom() {
  for (int i = 0; i < N_.size() - 1; ++i) {
    initRandom(i);
  }
}

void NeuralNet::initRandom(int i) {
  for (int j = 0; j < N_[i] + 1; ++j) {
    for ( int k = 0; k < N_[i + 1]; ++k) {
      double random = (2.0 * rand() / RAND_MAX) - 1;
//       w_[i](j, k) = random / sqrt(N_[i]);
      w_[i](j, k) = random;
    }
  }
}


void NeuralNet::subtractDeriv(double eps) {
  for (int i = 0; i < N_.size() - 1; ++i) {
    w_[i] -= eps * deriv_[i];
  }
}

void NeuralNet::doContrastiveDivergence(double eps) {
  // Re-adjust the first layer input so that their current value is interpreted as a probability
  for (int i = 0; i < N_[0] - 1; ++i) {
    a_[0][i] = sampleBernoulli(a_[0][i]);
  }
  
  calcActivationsWithBinaryRestriction();

  int layers = N_.size();

  // And do a step of CD
  for (int v = 0; v < N_[layers - 2]; ++v) {
    for (int h = 0; h < N_[layers - 1]; ++h) {
      
    }
  }
}

void NeuralNet::addLayer(int size) {
  N_.push_back(size);

  int layers = N_.size();
  w_.resize(layers - 1);
  w_[layers - 2].resize(N_[layers - 2] + 1, N_[layers - 1]);
  w_[layers - 2].setZero();
  deriv_ = w_;

  a_.resize(layers);
  a_[layers - 2].resize(N_[layers - 2] + 1);
  a_[layers - 2][N_[layers - 2]] = 1; // constant bias input
  a_[layers - 1].resize(N_[layers - 1]); // no need for constant bias input on last layer
  a_[layers - 1].setZero();
}
