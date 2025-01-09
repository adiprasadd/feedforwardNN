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
public:
    NeuralNetwork(int inputs, int hidden, int outputs) {
        srand(time(0));
        learning_rate = 0.1;

        input_layer = vector<double>(inputs);
        hidden_layer = vector<double>(hidden);
        output_layer = vector<double>(outputs);

        weights_ih = vector<vector<double>>(hidden, vector<double>(inputs));
        weights_ho = vector<vector<double>>(outputs, vector<double>(hidden));

        for(int i = 0; i < hidden; i++) {
            for(int j = 0; j < inputs; j++) {
                weights_ih[i][j] = random_weight();
            }
        }

        for(int i = 0; i < outputs; i++) {
            for(int j = 0; j < hidden; j++) {
                weights_ho[i][j] = random_weight();
            }
        }
    }

    vector<double> feedforward(vector<double> inputs) {
        for(int i = 0; i < input_layer.size(); i++) {
            input_layer[i] = inputs[i];
        }

        for(int i = 0; i < hidden_layer.size(); i++) {
            double sum = 0;
            for(int j = 0; j < input_layer.size(); j++) {
                sum += input_layer[j] * weights_ih[i][j];
            }
            hidden_layer[i] = sigmoid(sum);
        }

        for(int i = 0; i < output_layer.size(); i++) {
            double sum = 0;
            for(int j = 0; j < hidden_layer.size(); j++) {
                sum += hidden_layer[j] * weights_ho[i][j];
            }
            output_layer[i] = sigmoid(sum);
        }

        return output_layer;
    }

    void train(vector<double> inputs, vector<double> targets) {
        // Feedforward
        vector<double> outputs = feedforward(inputs);

        vector<double> output_errors(output_layer.size());
        for(int i = 0; i < output_layer.size(); i++) {
            double error = targets[i] - outputs[i];
            output_errors[i] = error * sigmoid_derivative(outputs[i]);
        }

        for(int i = 0; i < output_layer.size(); i++) {
            for(int j = 0; j < hidden_layer.size(); j++) {
                weights_ho[i][j] += learning_rate * output_errors[i] * hidden_layer[j];
            }
        }

        vector<double> hidden_errors(hidden_layer.size());
        for(int i = 0; i < hidden_layer.size(); i++) {
            double error = 0;
            for(int j = 0; j < output_layer.size(); j++) {
                error += output_errors[j] * weights_ho[j][i];
            }
            hidden_errors[i] = error * sigmoid_derivative(hidden_layer[i]);
        }

        for(int i = 0; i < hidden_layer.size(); i++) {
            for(int j = 0; j < input_layer.size(); j++) {
                weights_ih[i][j] += learning_rate * hidden_errors[i] * input_layer[j];
            }
        }
    }
};
