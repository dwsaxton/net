#include "worker.h"

#include <iostream>
using namespace std;

#include "mnist.h"

Worker::Worker()
{
}

Worker::~Worker()
{
}

void Worker::process() {
  // number of boxes in the different "layers"
  const int first_layer = 5;
  const int second_layer = 20;
  const int third_layer = 40;
  
  LayerParams layer0;
  layer0.connection_type = LayerParams::Initial;
  layer0.box_count = 1;
  layer0.box_edge = 29;
  
  LayerParams layer1;
  layer1.connection_type = LayerParams::Convolution;
  layer1.box_count = first_layer;
  layer1.box_edge = 25;
  layer1.mask_edge = 5;
  
  LayerParams layer2;
  layer2.connection_type = LayerParams::Pooling;
  layer2.box_count = first_layer;
  layer2.box_edge = 13;
  
  LayerParams layer3;
  layer3.connection_type = LayerParams::Convolution;
  layer3.box_count = second_layer;
  layer3.box_edge = 9;
  layer3.mask_edge = 5;
  
  LayerParams layer4;
  layer4.connection_type = LayerParams::Pooling;
  layer4.box_count = second_layer;
  layer4.box_edge = 5;
  
  LayerParams layer5;
  layer5.connection_type = LayerParams::Full;
  layer5.box_count = third_layer;
  layer5.box_edge = 1;
  
  LayerParams layer6;
  layer6.connection_type = LayerParams::Full;
  layer6.box_count = 10;
  layer6.box_edge = 1;
  
  LayerParams layer7;
  layer7.connection_type = LayerParams::SoftMax;
  layer7.box_count = 10;
  layer7.box_edge = 1;
  
  vector<LayerParams> params = {layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7};
  
//   LayerParams layer0;
//   layer0.connection_type = LayerParams::Initial;
//   layer0.box_count = 1;
//   layer0.box_edge = 28;
//   
//   LayerParams layer1;
//   layer1.connection_type = LayerParams::Full;
//   layer1.box_count = 15;
//   layer1.box_edge = 1;
//   
//   LayerParams layer2;
//   layer2.connection_type = LayerParams::Full;
//   layer2.box_count = 10;
//   layer2.box_edge = 1;
//   
//   vector<LayerParams> params = {layer0, layer1, layer2};
  
  net_ = new ConvNet(params);
  mnist_.init();

  int training_count = mnist_.trainingCount();
  
  float learning_rate = 0.05;
    
  int epoch = 0;
  
  while (true) {
    for (int i = 0; i < training_count; ++i) {
      if (i % 1000 == 0) {
        emit dataReady();
  //       cout << "Doing training " << i << "..." << endl;
      }
      Image image = mnist_.getTraining(rand() % training_count);
      net_->forwardPass(image.pixels);
      VectorXf target(10);
      if (i % 1000 == 0) {
        cout << "a=" << net_->getOutput().transpose() << " digit=" << image.digit << endl;
        cout << "a=" << net_->get2ndOutput().transpose() << " digit=" << image.digit << endl;
      }
      target.setZero();
      target[image.digit] = 1;
      net_->backwardsPass(target, learning_rate);
      
      if (i % 5000 == 0) {
        test();
      }
    }
    
    epoch++;
    if (epoch % 2 == 0) {
      learning_rate *= 0.5;
      if (learning_rate < 0.001) {
        learning_rate = 0.001;
      }
    }
//     cout << " epoch finished, new learning rate " << learning_rate << endl;
  }
}

static int test_number = 0;
void Worker::test() {
  int test_count = mnist_.testCount();
  
  int failing_count[10] = {0};
  
  int total = 0;
  for (int i = 0; i < test_count; ++i) {
    Image image = mnist_.getTraining(i);
    net_->forwardPass(image.pixels);
    VectorXf out = net_->getOutput();
    int target_digit = image.digit;
    float target_best = out(target_digit);
    bool good = true;
    for (int j = 0; j < 10; ++j) {
      if (target_digit != j && out(j) >= target_best) {
        good = false;
        break;
      }
    }
    if (good) {
      total++;
    } else {
      failing_count[target_digit]++;
      if (rand() % failing_count[target_digit] == 0) {
        failing[target_digit] = image;
      }
    }
  }
  
  cout << " Score: " << total << " (" << (100.0 * total / test_count) << "%)" << endl;
//   cout << 1.0 * total / test_count << endl;
  
//   if (++test_number == 100 ) {
//     exit(0);
//   }
}
