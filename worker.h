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
  
  MatrixXf sampleAutoencoder() const;

public Q_SLOTS:
  void process();

private:
  void doAutoencoder();
  void doDeep();
  void train(ConvNet *net, std::function<void ()> set_input_and_target, std::function<void ()> print_status, bool update_images);
  void test(ConvNet *net);
  
  Mnist mnist_;
  Interface *interface_;
};

#endif // WORKER_H
