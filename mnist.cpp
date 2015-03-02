#include "mnist.h"

Mnist::Mnist() {
  ifstream training_labels, training_pixels, test_labels, test_pixels;
  
  training_labels.open("/home/david/projects/Digit/train-labels-idx1-ubyte", ios::in | ios::binary | ios::ate);
  training_pixels.open("/home/david/projects/Digit/train-images-idx3-ubyte", ios::in | ios::binary | ios::ate);
  test_labels.open("/home/david/projects/Digit/t10k-labels-idx1-ubyte", ios::in | ios::binary | ios::ate);
  test_pixels.open("/home/david/projects/Digit/t10k-images-idx3-ubyte", ios::in | ios::binary | ios::ate);

  initImages(training_labels, training_pixels, 60000, training_);
  initImages(test_labels, test_pixels, 10000, test_);

  for (int i = 0; i < 10; ++i) {
    print(training_[i]);
  }
}

void Mnist::initImages(ifstream & labels, ifstream & pixels, int count, Image *images) {
  labels.seekg(8);
  pixels.seekg(16);
  Matrix<unsigned char, 28, 28> image;
  for (int i = 0; i < count; ++i) {
    pixels.read((char*) image.data(), 784);
    char label;
    labels.read(&label, 1);
    images[i].digit = label;
    images[i].pixels.resize(28, 28);
    images[i].pixels = image.cast<double>() / 255.0;
    
//     resize(images[i].pixels);
  }
}

void Mnist::resize(MatrixXd &pixels) const {
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
  
  if (first_i > 8) {
    first_i = 8;
  }
  if (first_j > 8) {
    first_j = 8;
  }
  
  assert(last_i - first_i + 1 <= 20);
  assert(last_j - first_j + 1 <= 20);
  
  pixels = pixels.block(first_i, first_j, 20, 20);
}

void Mnist::print(Image const& image) const {
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      double value = image.pixels(j, i);
      cout << (value > 0.5 ? "*" : " ");
    }
    cout << endl;
  }
}
