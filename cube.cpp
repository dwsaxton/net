#include "cube.h"

#include <utility>

inline void reorder(int d[], int stack_coordinate) {
  swap(d[0], d[stack_coordinate]);
}

Cube::Cube() {
  d0_ = 0;
  d1_ = 0;
  d2_ = 0;
  height_ = 0;
  rows_ = 0;
  cols_ = 0;
  stack_coordinate_ = 0;
}

Cube::Cube(int d0, int d1, int d2) {
  d0_ = d0;
  d1_ = d1;
  d2_ = d2;
  
  int d[3] = {d0, d1, d2};
  
  int min = d0;
  stack_coordinate_ = 0;
  for (int i = 1; i < 3; ++i) {
    if (d[i] < min) {
      min = d[i];
      stack_coordinate_ = i;
    }
  }
  
  reorder(d);
  init(d);
}

Cube::Cube(int d0, int d1, int d2, int stack_coordinate) {
  d0_ = d0;
  d1_ = d1;
  d2_ = d2;
  
  stack_coordinate_ = stack_coordinate;
 
  int d[3] = {d0, d1, d2};
  reorder(d);
  init(d);
}

void Cube::init(int * d) {
  height_ = d[0];
  rows_ = d[1];
  cols_ = d[2];
  
  data_.resize(height_);
  for (int i = 0; i < height_; ++i) {
    data_[i].resize(rows_, cols_);
  }
  
  setZero();
}
  
float Cube::computeKernel(Cube const& kernel, int i, int j) const {
  assert(kernel.stackCoordinate() == stackCoordinate());
  assert(kernel.d0_ == d0_);
  int s[3] = {0, i, j};
  reorder(s);
  
  float sum = 0;
  for (int a = 0; a < kernel.height_; ++a) {
    sum += data_[a + s[0]].block(s[1], s[2], kernel.rows_, kernel.cols_).cwiseProduct(kernel.data_[a]).sum();
  }
  return sum;
}

void Cube::addScaledKernel(float mult, Cube const& kernel, int i, int j) {
  assert(kernel.stackCoordinate() == stackCoordinate());
  assert(kernel.d0_ == d0_);
  int s[3] = {0, i, j};
  reorder(s);
  
  for (int a = 0; a < kernel.height_; ++a) {
    data_[a + s[0]].block(s[1], s[2], kernel.rows_, kernel.cols_) += mult * kernel.data_[a];
  }
}

void Cube::addScaledSubcube(float mult, Cube const& cube, int i, int j) {
  // TODO I Don't think these coordinates are coorreect1!
  assert(cube.stackCoordinate() == stackCoordinate());
  assert(cube.d0_ == d0_);
  int s[3] = {0, i, j};
  reorder(s);
  for (int a = 0; a < cube.height_; ++a) {
    data_[a] += mult * cube.data_[a + s[0]].block(s[1], s[2], rows_, cols_);
  }
}

float & Cube::operator()(int i, int j, int k) {
  int a[3] = {i, j, k};
  reorder(a);
  return data_[a[0]](a[1], a[2]);
}
float Cube::operator()(int i, int j, int k) const {
  int a[3] = {i, j, k};
  reorder(a);
  return data_[a[0]](a[1], a[2]);
}

void Cube::operator+=(Cube const& other) {
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  assert(height_ == other.height_);
  for (int i = 0; i < height_; ++i) {
    data_[i] += other.data_[i];
  }
}

void Cube::operator/=(float v) {
  for (int i = 0; i < height_; ++i) {
    data_[i] /= v;
  }
}

MatrixXf & Cube::layer(int i) {
  return data_[i];
}

MatrixXf const& Cube::layer(int i) const {
  return data_[i];
}

float Cube::squaredNorm() const {
  float sum = 0;
  for (int i = 0; i < height_; ++i) {
    sum += data_[i].squaredNorm();
  }
  return sum;
}

float Cube::maxCoeff() const {
  float max = -HUGE_VAL;
  for (int i = 0; i < height_; ++i) {
    max = std::max(max, data_[i].maxCoeff());
  }
  return max;
}

float Cube::minCoeff() const {
  float min = HUGE_VAL;
  for (int i = 0; i < height_; ++i) {
    min = std::min(min, data_[i].minCoeff());
  }
  return min;
}

void Cube::setZero() {
  for (int i = 0; i < height_; ++i) {
    data_[i].setZero();
  }
}

void Cube::setRandom() {
  for (int i = 0; i < height_; ++i) {
    data_[i].setRandom(); // (or = MatrixXf::random() ?)
  }
}

void test1() {
  Cube cube(1, 2, 2);
  
  cube(0, 0, 0) = 0;
  cube(0, 0, 1) = 1;
  cube(0, 1, 0) = 2;
  cube(0, 1, 1) = 3;
  
  Cube kernel(1, 1, 1);
  kernel(0, 0, 0) = 5;
  
  assert(cube.computeKernel(kernel, 1, 1) == 15);
}

void test2() {
  Cube kernel(3, 1, 1);
  kernel(0, 0, 0) = 5;
  kernel(1, 0, 0) = 7;
  kernel(2, 0, 0) = 11;
  
  Cube cube(3, 2, 2, kernel.stackCoordinate());
  
  cube(0, 0, 0) = 0;
  cube(0, 0, 1) = 1;
  cube(0, 1, 0) = 2;
  cube(0, 1, 1) = 3;
  
  cube(1, 0, 0) = 0;
  cube(1, 0, 1) = 1;
  cube(1, 1, 0) = 2;
  cube(1, 1, 1) = 3;
  
  cube(2, 0, 0) = 0;
  cube(2, 0, 1) = 1;
  cube(2, 1, 0) = 2;
  cube(2, 1, 1) = 3;
  
  assert(cube.computeKernel(kernel, 1, 1) == 15 + 21 + 33);
}

void TestCube() {
  test1();
  test2();
}
