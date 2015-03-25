#ifndef INTERFACE_H
#define INTERFACE_H

#include <vector>
using namespace std;

#include <QLabel>
#include <QWidget>

#include "ui_interface.h"

class ConvNet;
class Worker;

class Interface : public QWidget, public Ui::Interface {
  Q_OBJECT

public:
  Interface();
  
private slots:
  void updateImages(ConvNet*);

private:
  void showFirstLayer(ConvNet*);
  void showFailingSample();
  void showRandomTransformed();
  void showAutoencoded(ConvNet*);
  
  Worker *worker_;
  vector<QLabel*> images_;
  vector<QLabel*> layers_;
};

#endif // INTERFACE_H
