#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>

using namespace std;

class NeuralNetwork {
private:
    vector<double> input_layer;
    vector<double> hidden_layer;
    vector<double> output_layer;
    vector<vector<double>> weights_ih;
    vector<vector<double>> weights_ho;
    double learning_rate;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    double random_weight() {
        return ((double)rand() / RAND_MAX) * 2 - 1;
    }
