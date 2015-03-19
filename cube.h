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
  Cube();
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
  float operator()(int i, int j, int k) const;
  void operator+=(Cube const& other);
  void operator/=(float v);
  
  // TODO try to remove as many calls to layer as possible; in general we shouldn't be
  // allowing access to the internals, and should instead provide functions that do
  // whatever the outside uses are trying to do
  /**
   * @return the layer at coordinate @param i.
   */
  MatrixXf & layer(int i);
  MatrixXf const& layer(int i) const;
  
  /**
   * @return the convolution of @param kernel with this block, where the kernel is
   * offset to (i, j) (the first dimension offset is assumed to be zero). Using this
   * function requires the stack coordinate of the kernel to be the same as the stack
   * coordinate of this block.
   */
  float computeKernel(Cube const& kernel, int i, int j) const;
  /**
   * Adds the kernel to the subcube of this, like @see computeKernel and @see addScaledSubcube.
   */
  void addScaledKernel(float mult, Cube const& kernel, int i, int j);
  /**
   * Adds @param mult times the subcube of @param cube, where the subcube is located at
   * (0, i, j) (as in @see computeKernel), and is of dimensions the size of this cube.
   */
  void addScaledSubcube(float mult, Cube const& cube, int i, int j);
  
  float squaredNorm() const;
  float maxCoeff() const;
  float minCoeff() const;
  
  int d0() const { return d0_; }
  int d1() const { return d1_; }
  int d2() const { return d2_; }
  
  
  int height() const { return height_; }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  
  void setZero();
  void setRandom();
  
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
