#ifndef MNIST_H
#define MNIST_H

#include <fstream>
#include <iostream>
using namespace std;

#include <Eigen/Geometry>
using namespace Eigen;

struct Image {
  MatrixXd pixels; // pixels that make up the image
  int digit; // in range 0 to 9
};

class Mnist {
public:
  Mnist();

  int trainingCount() const { return 60000; }
  int testCount() const { return 10000; }
  Image getTraining(int i) const { return training_[i]; }
  Image getTest(int i) const { return test_[i]; }
  void print(Image const& image) const;

private:
  void initImages(ifstream& labels, ifstream& pixels, int count, Image* images);
  void resize(MatrixXd &pixels) const;

  Image training_[60000];
  Image test_[10000];
};

#endif // MNIST_H
