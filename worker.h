#ifndef WORKER_H
#define WORKER_H

#include <QObject>

#include "convnet.h"
#include "neuralnet.h"

class Mnist;

class Worker : public QObject
{
  Q_OBJECT

public:
  Worker();
  ~Worker();
  
  ConvNet *net_;
  Mnist *mnist_;
//   NeuralNet net_;

public slots:
  void process();

signals:
  void dataReady();
  
private:
  void test();
};

#endif // WORKER_H
