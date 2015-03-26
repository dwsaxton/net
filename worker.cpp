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
  
  float weight_decay = 0.0001;
  ConvNet *net = new ConvNet(createAutoencoder(400), weight_decay);
  
  auto set_input_and_target = [&] () {
    Image image = sampleRandomTraining();
    MatrixXf transformed = image.generate(RandomTransform(10, 0.1, 2.5));
    net->setInput(transformed);
    net->setTarget(transformed);
  };
  
  train(net, set_input_and_target);
  
  ConvNet *net2 = new ConvNet(createAutoencoderMiddle(400, 200), weight_decay);
  
  auto set_input_and_target_2 = [&] () {
    // TODO this
  };
  
  train(net2, set_input_and_target_2);
  
}

void Worker::train(ConvNet *net, std::function<void ()> set_input_and_target) {
  float learning_rate = 0.001;
  
  for (int done = 1; done < 100000; ++done) {
    set_input_and_target();
    net->forwardPass();
    net->backwardsPass(learning_rate);
    
    if (done % 100 == 0) {
      QCoreApplication::processEvents();
    }
    
    if (done % 2000 == 0) {
      interface_->updateImages(net);
    }
    
    if (done % 20000 == 0) {
      learning_rate *= 0.5;
      if (learning_rate < 1e-7) {
        learning_rate = 1e-7;
      }
      cout << "decreased learning rate to " << learning_rate << endl;
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
