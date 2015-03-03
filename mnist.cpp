#include "mnist.h"

#include <QColor>
#include <QImage>

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
    images[i].digit = label;
    images[i].pixels.resize(28, 28);
    images[i].pixels = image.cast<float>() / 255.0;
    
//     resize(images[i].pixels);
  }
}

void Mnist::resize(MatrixXf& pixels) const {
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
  
  // Sanity check that we're detecting non-zero pixels correctly, since all the
  // mnist digits should fit inside a 20 x 20 bounding box.
  int range_i = last_i - first_i + 1;
  int range_j = last_j - first_j + 1;
  assert(range_i <= 20);
  assert(range_j <= 20);
  
  MatrixXf block(range_i, range_j);
  block = pixels.block(first_i, first_j, range_i, range_j);
  pixels = block;
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

QImage Image::toQImage() {
  QImage image(pixels.rows(), pixels.cols(), QImage::Format_RGB32);
  for (int i = 0; i < pixels.rows(); ++i) {
    for (int j = 0; j < pixels.cols(); ++j) {
      int grey = (int) (255 * pixels(i, j));
      image.setPixel(i, j, QColor(grey, grey, grey).rgb());
    }
  }
  return image;
}
