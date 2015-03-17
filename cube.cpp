#include "cube.h"

#include <utility>

inline void reorder(int d[], int stack_coordinate) {
  swap(d[0], d[stack_coordinate]);
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

float & Cube::operator()(int i, int j, int k) {
  int a[3] = {i, j, k};
  reorder(a);
  return data_[a[0]](a[1], a[2]);
}

MatrixXf & Cube::layer(int i) {
  return data_[i];
}

void Cube::setZero() {
  for (int i = 0; i < height_; ++i) {
    data_[i].setZero();
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
