#include <QApplication>

#include <Eigen/Core>

#include "convnet.h"
#include "interface.h"

int main(int argc, char **argv) {
  TestConvNet();
  QApplication app(argc, argv);
  Interface interface;
  interface.show();
  return app.exec();
}
