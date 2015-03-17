#ifndef CUBE_H
#define CUBE_H

#include <utility>
#include <vector>
using namespace std;

#include <Eigen/Geometry>
using namespace Eigen;

void TestCube();

/**
 * 3-dimensional equivalent of a Vector / Matrix.
 * 
 * Internally uses a stack of MatrixXf.
 */
class Cube {
public:
  /**
   * Produces a Block of dimension d0 x d1 x d2, where the internal matrices
   * are stacked in the direction of the smallest of d0, d1, or d2
   */
  Cube(int d0, int d1, int d2);
  /**
   * Produces a Block whose internal matrices are stacked in direction
   * stack_coordinate.
   */
  Cube(int d0, int d1, int d2, int stack_coordinate);
  
  /**
   * @return the coordinate (1, 2, or 3) in which the internal matrices are stacked.
   */
  int stackCoordinate() const { return stack_coordinate_; }
  
  float & operator()(int i, int j, int k);
  
  /**
   * @return the layer at coordinate @param i.
   */
  MatrixXf & layer(int i);
  
  /**
   * @return the convolution of @param kernel with this block, where the kernel is
   * offset to (i, j) (the first dimension offset is assumed to be zero). Using this
   * function requires the stack coordinate of the kernel to be the same as the stack
   * coordinate of this block.
   */
  float computeKernel(Cube const& kernel, int i, int j) const;
  
  int d0() const { return d0_; }
  int d1() const { return d1_; }
  int d2() const { return d2_; }
  
  
  int height() const { return height_; }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  
private:
  void reorder(int *d) const {
    swap(d[0], d[stack_coordinate_]);
  }
  void init(int * d);
  
  int stack_coordinate_;
  
  int height_;
  int rows_;
  int cols_;
  
  int d0_;
  int d1_;
  int d2_;
  
  vector<MatrixXf> data_;
};

#endif // CUBE_H
