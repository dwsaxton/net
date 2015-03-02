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
  Worker *worker;
  vector<QLabel*> images_;
};

#endif // INTERFACE_H
