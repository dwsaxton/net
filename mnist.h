#ifndef MNIST_H
#define MNIST_H

#include <fstream>
#include <iostream>
using namespace std;

#include <Eigen/Geometry>
using namespace Eigen;

class RandomTransform;

class QImage;
class QPoint;
class QPointF;

const int TRAINING_COUNT = 60000;
const int TEST_COUNT = 10000;

class Image {
public:
  int digit() const { return digit_; }
  MatrixXf original() const { return pixels_orig_; }
  MatrixXf generate(RandomTransform const& transform) const;
  
  void set(MatrixXf const& pixels, int digit);
  
private:
  MatrixXf pixels_orig_; // pixels that make up the image in full 28 x 28 image
  MatrixXf pixels_cropped_; // minimal bounding box of the above pixels
  int digit_; // in range 0 to 9
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

class GaussianBlur {
public:
  GaussianBlur(float std_dev);
  
  MatrixXf apply(MatrixXf const& input) const;
  
private:
  VectorXf coeffs_;
};

/**
 * Elastic transform for a 28 x 28 image.
 */
class ElasticTransform {
public:
  ElasticTransform(float eps, GaussianBlur const& blur);
  
  QPointF map(QPoint const& input) const;
  
private:
  MatrixXf delta_x_;
  MatrixXf delta_y_;
};

/**
 * Applies the matrix
 * 
 *  ( 1 + a    b   )
 *  (   c    1 + d )
 * 
 * where a, b, c, d ~ N(0, eps).
 */
class AffineTransform {
public:
  AffineTransform(float eps);
  
  QPointF map(QPointF const& input) const;
  
private:
  float a_;
  float b_;
  float c_;
  float d_;
};

class RandomTransform {
public:
  RandomTransform(float eps_elastic, float eps_affine, float elastic_blur_std_dev);
  
  QPointF map(QPoint const& input) const;
  
private:
  ElasticTransform const elastic_;
  AffineTransform const affine_;
};

class Interpolator {
public:
  Interpolator(RandomTransform const& transform);
  
  MatrixXf apply(MatrixXf const& input) const;
  
private:
  RandomTransform transform)_;

#endif // MNIST_H
