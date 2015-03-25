#ifndef INTERFACE_H
#define INTERFACE_H

#include <vector>
using namespace std;

#include <QLabel>
#include <QWidget>

#include "ui_interface.h"

class Worker;

class Interface : public QWidget, public Ui::Interface {
  Q_OBJECT

public:
  Interface();
  
private slots:
  void updateImages();

private:
  void showFirstLayer();
  void showFailingSample();
  void showRandomTransformed();
  void showAutoencoded();
  
  Worker *worker_;
  vector<QLabel*> images_;
  vector<QLabel*> layers_;
};

#endif // INTERFACE_H
