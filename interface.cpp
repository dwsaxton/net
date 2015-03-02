#include "interface.h"
#include "ui_interface.h"

#include <iostream>
using namespace std;

#include <QThread>

#include "worker.h"

Interface::Interface()
{
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

  QThread *thread = new QThread;
  worker = new Worker;
  worker->moveToThread(thread);
  connect(thread, SIGNAL(started()), worker, SLOT(process()));
  connect(worker, SIGNAL(dataReady()), this, SLOT(updateImages()));
  thread->start();
}

void Interface::updateImages() {
  for (int i = 0; i < 5; ++i) {
    QImage image(5, 5, QImage::Format_RGB32);
    for (int x = 0; x < 5; ++x) {
      for (int y = 0; y < 5; ++y) {
        double value = worker->net_->layers_[1].boxes[i].weights.mask[0](x, y);
        if (value < -1) {
          value = -1;
        } else if (value > 1) {
          value = 1;
        }
        unsigned char grey = (1 - value) * 127;
        image.setPixel(x, y, QColor(grey, grey, grey).rgb());
      }
    }
    images_[i]->setPixmap(QPixmap::fromImage(image));
  }
}
      
//   for (int i = 0; i < 15; ++i) {
//     QImage image(20, 20, QImage::Format_RGB32);
//     for (int j = 0; j < 20; ++j) {
//       for (int k = 0; k < 20; ++k) {
//         double value = worker->net_->layers_[1].boxes[i].weights.mask[0](j, k);
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
// }
