#include "interface.h"
#include "ui_interface.h"

#include <iostream>
using namespace std;

#include <QPainter>
#include <QThread>

#include "cube.h"
#include "worker.h"

Interface::Interface()
{
  TestCube();
  
  setupUi(this);

  images_.resize(15);
  images_[0] = image00;
  images_[1] = image01;
  images_[2] = image02;
  images_[3] = image03;
  images_[4] = image04;
  images_[5] = image05;
  images_[6] = image06;
  images_[7] = image07;
  images_[8] = image08;
  images_[9] = image09;
  images_[10] = image10;
  images_[11] = image11;
  images_[12] = image12;
  images_[13] = image13;
  images_[14] = image14;
  
  layers_.resize(4);
  layers_[0] = layer0;
  layers_[1] = layer1;
  layers_[2] = layer2;
  layers_[3] = layer3;
  
//   showRandomTransformed();
  
  QThread *thread = new QThread;
  worker_ = new Worker;
  worker_->moveToThread(thread);
  connect(thread, SIGNAL(started()), worker_, SLOT(process()));
  connect(worker_, SIGNAL(dataReady()), this, SLOT(updateImages()));
  thread->start();
}

QImage toQImage(MatrixXf values, float min = 0, float max = 1) {
  int rows = values.rows();
  int cols = values.cols();
  QImage image(rows, cols, QImage::Format_RGB32);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float v = values(i, j);
      if (max == min) {
//         cout << v
        v = 0;
      } else {
        v = (v - min) / (max - min);
      }
//       cout << "v=" << v << endl;
      int grey = (int) (255 * v);
      image.setPixel(i, j, QColor(grey, grey, grey).rgb());
    }
  }
  return image;
}

void Interface::showRandomTransformed() {
  Mnist mnist;
  mnist.init();
  
  Image image = mnist.getTraining(random() % mnist.trainingCount());
  
  for (int i = 0; i < 15; ++i) {
    RandomTransform transform(10, 0.15, 2.5);
    MatrixXf mapped = image.generate(transform);
    images_[i]->setPixmap(QPixmap::fromImage(toQImage(mapped, 0, 1)));
  }
}

void Interface::updateImages() {
  for (int i = 0; i < 5; ++i) {
    QImage image(5, 5, QImage::Format_RGB32);
    for (int x = 0; x < 5; ++x) {
      for (int y = 0; y < 5; ++y) {
        double value = worker_->net_->layers_[1].kernels[i].cube(0, x, y);
        if (value < -1) {
          value = -1;
        } else if (value > 1) {
          value = 1;
        }
        unsigned char grey = (1 - value) * 127;
        image.setPixel(x, y, QColor(grey, grey, grey).rgb());
      }
    }
    images_[i]->setPixmap(QPixmap::fromImage(image.scaled(100, 100)));
  }
  for (int i = 0; i < 10; ++i) {
    QImage image = toQImage(worker_->failing[i].original());
    images_[i+5]->setPixmap(QPixmap::fromImage(image));
  }

  
//   Image image = worker_->sampleRandomTraining();
//   layers_[0]->setPixmap(QPixmap::fromImage(toQImage(image.original())));
//   worker_->net_->forwardPass(image.original());
//   
//   int layers[3] = {2, 4, 5};
//   int col_count[3] = {5, 10, 20};
//   int row_count[3] = {1, 2, 2};
//   int box_image_size[3] = {100, 50, 25};
//   
//   for (int layer_index = 0; layer_index < 3; ++layer_index) {
//     QImage image(500, row_count[layer_index] * box_image_size[layer_index], QImage::Format_RGB32);
//     QPainter painter(&image);
//     
//     Layer const & layer = worker_->net_->layers_[layers[layer_index]];
//     int features = layer.features();
//     
//     float min = layer.value.minCoeff();
//     float max = layer.value.maxCoeff();
//     
//     for (int box = 0; box < box_count; ++box) {
//       QImage box_image = toQImage(layer.boxes[box].values, min, max);
//       int edge = box_image_size[i];
//       QRect dest_pos(edge * (box % col_count[i]), edge * (box / col_count[i]), edge, edge);
//       painter.drawImage(dest_pos, box_image);
//     }
//     
//     layers_[i+1]->setPixmap(QPixmap::fromImage(image));
//   }
  
  
  
//   for (int i = 0; i < 15; ++i) {
//     Cube const& kernel = worker_->net_->layers_[1].kernels[i].cube;
//     QImage image(28, 28, QImage::Format_RGB32);
//     for (int j = 0; j < 28; ++j) {
//       for (int k = 0; k < 28; ++k) {
//         double value = kernel(0, j, k);
//         if (value < -1) {
//           value = -1;
//         } else if (value > 1) {
//           value = 1;
//         }
//         unsigned char grey = (1 - value) * 127;
//         image.setPixel(j, k, QColor(grey, grey, grey).rgb());
//       }
//     }
//     images_[i]->setPixmap(QPixmap::fromImage(image));
//   }
}
