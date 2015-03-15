#include "mnist.h"

#include <random>

#include <QColor>
#include <QImage>
#include <QPointF>

Mnist::Mnist() {
}

void Mnist::init() {
  training_.resize(TRAINING_COUNT);
  test_.resize(TEST_COUNT);
  
  ifstream training_labels, training_pixels, test_labels, test_pixels;
  
  training_labels.open("/home/david/projects/Digit/train-labels-idx1-ubyte", ios::in | ios::binary | ios::ate);
  training_pixels.open("/home/david/projects/Digit/train-images-idx3-ubyte", ios::in | ios::binary | ios::ate);
  test_labels.open("/home/david/projects/Digit/t10k-labels-idx1-ubyte", ios::in | ios::binary | ios::ate);
  test_pixels.open("/home/david/projects/Digit/t10k-images-idx3-ubyte", ios::in | ios::binary | ios::ate);

  initImages(training_labels, training_pixels, TRAINING_COUNT, training_);
  initImages(test_labels, test_pixels, TEST_COUNT, test_);

  for (int i = 0; i < 10; ++i) {
    print(training_[i]);
  }
}

void Mnist::initImages(ifstream & labels, ifstream & pixels, int count, vector<Image> &images) {
  labels.seekg(8);
  pixels.seekg(16);
  Matrix<unsigned char, 28, 28> image;
  for (int i = 0; i < count; ++i) {
    pixels.read((char*) image.data(), 784);
    char label;
    labels.read(&label, 1);
    images[i].set(label, image.cast<float>() / 255.0);
  }
}

void Mnist::print(Image const& image) const {
  for (int i = 0; i < image.pixels.cols(); ++i) {
    for (int j = 0; j < image.pixels.rows(); ++j) {
      float value = image.pixels(j, i);
      cout << (value > 0.5 ? "*" : " ");
    }
    cout << endl;
  }
}

void Image::set(MatrixXf const& pixels, int digit) {
  digit_ = digit;
  pixels_orig_ = pixels;
  
  assert(pixels.rows() == 28);
  assert(pixels.cols() == 28);
  int first_i = 27;
  int last_i = 0;
  int first_j = 27;
  int last_j = 0;
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      if (pixels(i, j) != 0) {
        if (i < first_i) {
          first_i = i;
        }
        if (i > last_i) {
          last_i = i;
        }
        if (j < first_j) {
          first_j = j;
        }
        if (j > last_j) {
          last_j = j;
        }
      }
    }
  }
  
  int range_i = last_i - first_i + 1;
  int range_j = last_j - first_j + 1;
  // Sanity check that we're detecting non-zero pixels correctly, since all the
  // mnist digits should fit inside a 20 x 20 bounding box.
  assert(range_i <= 20);
  assert(range_j <= 20);
  
  pixels_cropped_ = pixels.block(first_i, first_j, range_i, range_j);
}

std::default_random_engine const generator;

float safeGet(MatrixXf const& input, int i, int j) const {
  if (i < 0 || i >= input.rows() || j < 0 || j >= input.cols()) {
    return 0;
  }
  return input(i, j);
}

GaussianBlur::GaussianBlur(float std_dev) {
  int count = (int) ceil(6 * std_dev);
  if (count % 2 == 0) {
    count++;
  }
  
  coeffs_.resize(count);
  int mid = (count - 1) / 2;
  for (int i = 0; i <= mid; ++i) {
    float value = exp(- i * i / 2.0 / std_dev / std_dev);
    coeffs_(mid - i) = coeffs_(mid + i) = value;
  }
  
  coeffs_ /= coeffs_.sum();
}
  
MatrixXf GaussianBlur::apply(MatrixXf const& m0) const {
  int r = input.rows();
  int c = input.cols();
  int mid = (coeffs_.size() - 1) / 2;
  
  MatrixXf m1(r, c);
  m1.setZero();
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = -mid; k <= mid; ++k) {
        m1(i, j) += coeffs_(k + mid) * safeGet(m0, i + k, j);
      }
    }
  }
  
  MatrixXf m2(r, c);
  m2.setZero();
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = -mid; k <= mid; ++k) {
        m2(i, j) += coeffs_(k + mid) * safeGet(m1, i, j + k);
      }
    }
  }
  
  return m2;
}

RandomTransform::RandomTransform(float eps_elastic, float eps_affine, float elastic_blur_std_dev) 
    : elastic_(eps_elastic, GaussianBlur(elastic_blur_std_dev)),
      affine_(eps_affine) {
}
  
QPointF RandomTransform::map(QPoint const& input) const {
  return affine_.map(elastic_.map(input) - QPoint(14, 14)) + QPoint(14, 14);
}

AffineTransform::AffineTransform(float eps) {
  assert(eps > 0 && eps < 1); // sanity check
  std::normal_distribution<float> dist(0, eps);
  
  a_ = dist(generator);
  b_ = dist(generator);
  c_ = dist(generator);
  d_ = dist(generator);
}

QPointF AffineTransform::map(QPointF const& input) const {
  float x = input.x();
  float y = input.y();
  return QPointF((1 + a_) * x + b_ * y, c_ * x + (1 + d_) * y);
}

ElasticTransform::ElasticTransform(float eps, GaussianBlur const& blur) {
  delta_x_.resize(28, 28);
  delta_y_.resize(28, 28);
  
  std::normal_distribution<float> dist(0, eps);
  
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      delta_x_(i, j) = dist(generator);
      delta_y_(i, j) = dist(generator);
    }
  }
  
  delta_x_ = blur.apply(delta_x_);
  delta_y_ = blur.apply(delta_y_);
}

QPointF ElasticTransform::map(QPoint const& input) const {
  return QPointF(delta_x_[input.x()], delta_y_[input.y()]) + input;
}

Interpolator::Interpolator(RandomTransform const& transform)
    : transform_(transform) {
}
  
MatrixXf Interpolator::apply(MatrixXf const& input) const {
  MatrixXf output(28, 28);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      QPointF pixel = transform_.map(QPoint(i, j));
      int a = (int) pixel.x();
      int b = (int) pixel.y();
      float s = pixel.x() - a;
      float t = pixel.y() - b;
      
      output(i, j) =
          (1 - s) * (1 - t) * safeGet(input, a, b)
          + s * (1 - t) * safeGet(input, a + 1, b)
          + (1 - s) * t * safeGet(input, a, b + 1)
          + s * t * safeGet(input, a + 1, b + 1);
    }
  }
  return output;   
}

