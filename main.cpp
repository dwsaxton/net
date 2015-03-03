#include <QApplication>

#include <Eigen/Core>

#include "interface.h"

int main(int argc, char **argv) {
  Eigen::initParallel();
  QApplication app(argc, argv);
  Interface interface;
  interface.show();
  return app.exec();
}
