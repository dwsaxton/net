#ifndef MNIST_H
#define MNIST_H

#include <fstream>
#include <iostream>
using namespace std;

#include <Eigen/Geometry>
using namespace Eigen;

class QImage;

const int TRAINING_COUNT = 60000;
const int TEST_COUNT = 10000;

class Image {
public:
  MatrixXf pixels; // pixels that make up the image
  int digit; // in range 0 to 9
  QImage toQImage();
};

class Mnist {
public:
  Mnist();
  
  void init();

  int trainingCount() const { return TRAINING_COUNT; }
  int testCount() const { return TEST_COUNT; }
  Image getTraining(int i) const { return training_[i]; }
  Image getTest(int i) const { return test_[i]; }
  void print(Image const& image) const;

private:
  void initImages(ifstream& labels, ifstream& pixels, int count, vector<Image> & images);
  void resize(MatrixXf &pixels) const;

  vector<Image> training_;
  vector<Image> test_;
};

#endif // MNIST_H
