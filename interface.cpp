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

  images_.resize(25);
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
  images_[15] = image15;
  images_[16] = image16;
  images_[17] = image17;
  images_[18] = image18;
  images_[19] = image19;
  images_[20] = image20;
  images_[21] = image21;
  images_[22] = image22;
  images_[23] = image23;
  images_[24] = image24;
  
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
      if (v < 0 || !isfinite(v)) {
        v = 0;
      } else if (v > 1) {
        v = 1;
      }
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
  
  for (int i = 0; i < 25; ++i) {
    RandomTransform transform(10, 0.15, 2.5);
    MatrixXf mapped = image.generate(transform);
    images_[i]->setPixmap(QPixmap::fromImage(toQImage(mapped, 0, 1)));
  }
}

void Interface::showFirstLayer() {
  Layer const& layer = worker_->net_->layers_[1];
  float mn = HUGE_VAL;
  float mx = -HUGE_VAL;
  
  int features = layer.kernels.size();
  
  for (int i = 0; i < features; ++i) {
    mn = min(mn, layer.kernels[i].cube.minCoeff());
    mx = max(mx, layer.kernels[i].cube.maxCoeff());
  }
  
  for (int i = 0; i < features; ++i) {
    QImage image = toQImage(layer.kernels[i].cube.layer(0), mn, mx);
    images_[i]->setPixmap(QPixmap::fromImage(image.scaled(100, 100)));
  }
}

void Interface::showFailingSample() {
  for (int i = 0; i < 10; ++i) {
    QImage image = toQImage(worker_->failing[i].original());
    images_[i+15]->setPixmap(QPixmap::fromImage(image));
  }
}

void Interface::showAutoencoded() {
  for (int i = 0; i < 10; ++i) {
    int at = i + (i >= 5 ? 5 : 0);
    Image image = worker_->sampleRandomTraining();
    images_[at]->setPixmap(QPixmap::fromImage(toQImage(image.original())));
    worker_->net_->forwardPass(image.original());
    VectorXf output = worker_->net_->getOutput();
    assert(output.size() == 28 * 28);
    MatrixXf output_as_square(28, 28);
    for (int i = 0; i < 28 * 28; ++i) {
      output_as_square(i / 28, i % 28) = output(i);
    }
    images_[at + 5]->setPixmap(QPixmap::fromImage(toQImage(output_as_square)));
  }
}


void Interface::updateImages() {
//   showFirstLayer();
//   showFailingSample();
  showAutoencoded();
  
  
  
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
  
  
  
}
