#ifndef WORKER_H
#define WORKER_H

#include <QObject>

#include "convnet.h"
#include "mnist.h"
#include "neuralnet.h"

class Worker : public QObject
{
  Q_OBJECT

public:
  Worker();
  ~Worker();
  
  ConvNet *net_;
  
  Image failing[10];
  
//   NeuralNet net_;

public slots:
  void process();

signals:
  void dataReady();
  
private:
  void test();
  
  Mnist mnist_;
};

#endif // WORKER_H
