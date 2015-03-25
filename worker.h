#ifndef WORKER_H
#define WORKER_H

#include <QObject>

#include "mnist.h"

class ConvNet;

class Worker : public QObject
{
  Q_OBJECT

public:
  Worker();
  ~Worker();
  
  Image failing[10];
  
  Image sampleRandomTraining() const;

public slots:
  void process();

signals:
  void dataReady(ConvNet *net);
  
private:
  void train(ConvNet *net, std::function<void ()> set_input_and_target);
  void test(ConvNet *net);
  
  Mnist mnist_;
};

#endif // WORKER_H
