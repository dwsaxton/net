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

bool isCorrect(VectorXf const& out, int target) {
  float target_best = out(target);
  for (int j = 0; j < 10; ++j) {
    if (target != j && out(j) >= target_best) {
      return false;
    }
  }
  return true;
}

void Worker::process() {
  // number of boxes in the different "layers"
  const int first_layer = 5;
  const int second_layer = 20;
  const int third_layer = 40;
  
  LayerParams layer0;
  layer0.connection_type = LayerParams::Initial;
  layer0.features = 1;
  layer0.edge = 29;
  
  LayerParams layer1;
  layer1.connection_type = LayerParams::Convolution;
  layer1.features = first_layer;
  layer1.edge = 13;
  layer1.kernel = 5;
  layer1.stride = 2;
  
  LayerParams layer2;
  layer2.connection_type = LayerParams::Convolution;
  layer2.features = second_layer;
  layer2.edge = 5;
  layer2.kernel = 5;
  layer2.stride = 2;
  
  LayerParams layer3;
  layer3.connection_type = LayerParams::Convolution;
  layer3.features = third_layer;
  layer3.edge = 1;
  layer3.kernel = 5;
  
  LayerParams layer4;
  layer4.connection_type = LayerParams::Convolution;
  layer4.features = 10;
  layer4.edge = 1;
  
  LayerParams layer5;
  layer5.connection_type = LayerParams::SoftMax;
  layer5.features = 10;
  layer5.edge = 1;
  
  vector<LayerParams> params = {layer0, layer1, layer2, layer3, layer4, layer5};
  
//   LayerParams layer0;
//   layer0.connection_type = LayerParams::Initial;
//   layer0.features = 1;
//   layer0.edge = 28;
//   
//   LayerParams layer1;
//   layer1.connection_type = LayerParams::Convolution;
//   layer1.features = 15;
//   layer1.edge = 1;
//   layer1.kernel = 28;
//   
//   LayerParams layer2;
//   layer2.connection_type = LayerParams::Convolution;
//   layer2.features= 10;
//   layer2.edge = 1;
//   
//   LayerParams layer3;
//   layer3.connection_type = LayerParams::SoftMax;
//   layer3.features= 10;
//   layer3.edge = 1;
//   
//   vector<LayerParams> params = {layer0, layer1, layer2, layer3};
  
  net_ = new ConvNet(params);
  mnist_.init();
  
  bool trailing_correct[1000] = {false};
  int trailing_at = 0;
  int trailing_count = 0;
  int done = 0;
  
  while (true) {
    RandomTransform transform(10, 0.15, 2.5);
    Image image = sampleRandomTraining();
    VectorXf target(10);
    target.setZero();
    target[image.digit()] = 1;
    net_->forwardPass(image.generate(transform));
    net_->backwardsPass(target);
    
    bool correct = isCorrect(net_->getOutput(), image.digit());
    if (trailing_correct[trailing_at]) {
      trailing_count --;
    }
    trailing_correct[trailing_at] = correct;
    if (correct) {
      trailing_count++;
    }
    trailing_at = (trailing_at + 1) % 1000;
    
    if (done++ % 5000 == 0) {
      cout << "a=" << net_->getOutput().transpose() << " digit=" << image.digit() << endl;
      cout << "trailing=" << (trailing_count / 10.0) << "%" << endl;
//         cout << "2=" << net_->get2ndOutput().transpose() << " digit=" << image.digit << endl;
      test();
      emit dataReady();
    }
  }
}

Image Worker::sampleRandomTraining() const {
  return mnist_.getTraining(rand() % mnist_.trainingCount());
}

static int test_number = 0;
void Worker::test() {
  int test_count = mnist_.testCount();
  
  int failing_count[10] = {0};
  
  int total = 0;
  for (int i = 0; i < test_count; ++i) {
    Image image = mnist_.getTraining(i);
    net_->forwardPass(image.original());
    VectorXf out = net_->getOutput();
    int target_digit = image.digit();
    bool good = isCorrect(out, target_digit);
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
