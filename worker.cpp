#include "worker.h"

#include <iostream>
using namespace std;

#include "convnet.h"
#include "interface.h"
#include "mnist.h"

Worker::Worker(Interface *interface) {
  interface_ = interface;
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

vector<LayerParams> createDeepMnist() {
  // number of boxes in the different "layers"
  const int first_layer = 5;
  const int second_layer = 50;
  const int third_layer = 100;
  
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
  layer5.connection_type = LayerParams::Scale;
  layer5.features = 10;
  layer5.edge = 1;
  layer5.scale = 7;
  
  LayerParams layer6;
  layer6.connection_type = LayerParams::SoftMax;
  layer6.features = 10;
  layer6.edge = 1;
  
  return vector<LayerParams>({layer0, layer1, layer2, layer3, layer4, layer5, layer6});
}

vector<LayerParams> createShallowMnist() {
  LayerParams layer0;
  layer0.connection_type = LayerParams::Initial;
  layer0.features = 1;
  layer0.edge = 28;
  
  LayerParams layer1;
  layer1.connection_type = LayerParams::Convolution;
  layer1.features = 15;
  layer1.edge = 1;
  layer1.kernel = 28;
  
  LayerParams layer2;
  layer2.connection_type = LayerParams::Convolution;
  layer2.features= 10;
  layer2.edge = 1;
  
  LayerParams layer3;
  layer3.connection_type = LayerParams::Scale;
  layer3.features= 10;
  layer3.edge = 1;
  layer3.scale = 7;
  
  LayerParams layer4;
  layer4.connection_type = LayerParams::SoftMax;
  layer4.features= 10;
  layer4.edge = 1;
  
  return vector<LayerParams>({layer0, layer1, layer2, layer3, layer4});
}

vector<LayerParams> createAutoencoder(int middle_features) {
  LayerParams layer0;
  layer0.connection_type = LayerParams::Initial;
  layer0.edge = 28;
  
  LayerParams layer1;
  layer1.connection_type = LayerParams::Convolution;
  layer1.features = middle_features;
  layer1.kernel = 28;
  layer1.neuron_type = LayerParams::Sigmoid;
  
  LayerParams layer2;
  layer2.connection_type = LayerParams::Convolution;
  layer2.features = 28 * 28;
  layer2.neuron_type = LayerParams::Sigmoid;
  
//   LayerParams layer3;
//   layer3.connection_type = LayerParams::Scale;
//   layer3.features= 28 * 28;
  
  return vector<LayerParams>({layer0, layer1, layer2});
}

vector<LayerParams> createAutoencoderMiddle(int input_features, int middle_features) {
  LayerParams layer0;
  layer0.connection_type = LayerParams::Initial;
  layer0.features = input_features;
  
  LayerParams layer1;
  layer1.connection_type = LayerParams::Convolution;
  layer1.features = middle_features;
  layer1.neuron_type = LayerParams::Sigmoid;
  
  LayerParams layer2;
  layer2.connection_type = LayerParams::Convolution;
  layer2.features = input_features;
  layer2.neuron_type = LayerParams::Sigmoid;
  
  return vector<LayerParams>({layer0, layer1, layer2});
}

void Worker::process() {
  mnist_.init();
  
  doAutoencoder();
//   doDeep();
}

void Worker::doDeep() {
  float weight_decay = 0.0001;
  ConvNet *net = new ConvNet(createDeepMnist(), weight_decay);
  
  auto set_input_and_target = [&] () {
    Image image = sampleRandomTraining();
    MatrixXf transformed = image.generate(RandomTransform(10, 0.1, 2.5));
    net->setInput(transformed);
    net->setTarget(image.digit());
  };
  
  auto print_status = [&] () {
    cout << net->getOutput().transpose() << " t=" << net->getTargetInt() << endl;
    cout << net->getOutput2().transpose() << endl;
  };
  
  train(net, set_input_and_target, print_status, true);
}

MatrixXf Worker::sampleAutoencoder() const {;
  MatrixXf input(28, 28);
  input.setZero();
  int a = rand() % 24;
  int b = rand() % 24;
  int c = rand() % 24;
  int d = rand() % 24;
  input.block(min(a, b), min(c, d), abs(b - a) + 4, abs(d - c) + 4).setConstant(1);
  return input;
}
  

void Worker::doAutoencoder() {
  float weight_decay = 0.0001;
  const int first = 100;
  const int second = 50;
  const int third = 25;
  ConvNet *net = new ConvNet(createAutoencoder(first), weight_decay);
  
  auto set_input_and_target = [&] () {
    MatrixXf target = sampleRandomTraining().generate(RandomTransform(10, 0.1, 2.5));
//     MatrixXf target = sampleAutoencoder();
    
    MatrixXf noisy = target + 0.5 * MatrixXf::Random(28, 28);
    net->setInput(noisy);
    net->setTarget(target);
  };
  
  auto print_status = [] () {
  };
  
  train(net, set_input_and_target, print_status, true);
  
  ConvNet *net2 = new ConvNet(createAutoencoderMiddle(first, second), weight_decay);
  auto set_input_and_target_2 = [&] () {
    MatrixXf outer = sampleRandomTraining().generate(RandomTransform(10, 0.1, 2.5));
    net->setInput(outer);
    net->forwardPass();
    VectorXf middle = net->getOutput2();
    VectorXf noisy = middle + 0.5 * VectorXf::Random(first);
    net2->setInput(noisy);
    net2->setTarget(middle);
  };
  train(net2, set_input_and_target_2, print_status, false);
  
  net->layers_.insert(net->layers_.begin() + 2, net2->layers_[1]);
  net->layers_.insert(net->layers_.begin() + 3, net2->layers_[2]);
  train(net, set_input_and_target, print_status, true);
  
  ConvNet *net3 = new ConvNet(createAutoencoderMiddle(second, third), weight_decay);
  auto set_input_and_target_3 = [&] () {
    MatrixXf outer = sampleRandomTraining().generate(RandomTransform(10, 0.1, 2.5));
    net->setInput(outer);
    net->forwardPass();
    VectorXf middle = net->getOutputPrior(2);
    VectorXf noisy = middle + 0.5 * VectorXf::Random(first);
    net3->setInput(noisy);
    net3->setTarget(middle);
  };
  train(net3, set_input_and_target_3, print_status, false);
  
  net->layers_.insert(net->layers_.begin() + 3, net3->layers_[1]);
  net->layers_.insert(net->layers_.begin() + 4, net3->layers_[2]);
  train(net, set_input_and_target, print_status, true);
}

void Worker::train(ConvNet *net, std::function<void ()> set_input_and_target, std::function<void ()> print_status, bool update_images) {
  float learning_rate = 0.01;
  float leak = 0.01;
  setLeak(leak);
  
  for (int done = 1; done < 100000; ++done) {
    set_input_and_target();
    net->forwardPass();
    net->backwardsPass(learning_rate);
    
    if (done % 100 == 0) {
      QCoreApplication::processEvents();
    }
    
    if (done % 2000 == 0 && update_images) {
      interface_->updateImages(net);
    }
    
    if (done % 5000 == 0) {
//       test(net);
      print_status();
    }
    
    if (done % 10000 == 0) {
      learning_rate *= 0.5;
      if (learning_rate < 1e-7) {
        learning_rate = 1e-7;
      }
      cout << "decreased learning rate to " << learning_rate << endl;
    }
    
    if (done % 2000 == 0 && leak > 0) {
      leak *= 0.7;
      if (leak < 1e-8) {
        leak = 0;
      }
      setLeak(leak);
      cout << "decreased leak to " << leak << endl;
    }
  }
}

Image Worker::sampleRandomTraining() const {
  return mnist_.getTraining(rand() % mnist_.trainingCount());
}

static int test_number = 0;

void Worker::test(ConvNet *net) {
  int test_count = mnist_.testCount();
  
  int failing_count[10] = {0};
  
  int total = 0;
  for (int i = 0; i < test_count; ++i) {
    Image image = mnist_.getTraining(i);
    net->setInput(image.original());
    net->forwardPass();
    VectorXf out = net->getOutput();
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
}
