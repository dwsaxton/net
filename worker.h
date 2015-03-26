#ifndef WORKER_H
#define WORKER_H

#include <QObject>

#include "mnist.h"

class ConvNet;
class Interface;

class Worker : public QObject
{
  Q_OBJECT

public:
  Worker(Interface *interface);
  ~Worker();
  
  Image failing[10];
  
  Image sampleRandomTraining() const;

public Q_SLOTS:
  void process();

private:
  void train(ConvNet *net, std::function<void ()> set_input_and_target);
  void test(ConvNet *net);
  
  Mnist mnist_;
  Interface *interface_;
};

#endif // WORKER_H
