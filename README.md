# Feedforward Neural Network for XOR Problem  

This repository contains an implementation of a feedforward neural network implemented in C++ only using standard libraries, with a single hidden layer. The network is designed to solve the XOR problem, a classic binary classification task. The network is trained using backpropagation and employs the sigmoid activation function.  

## Features  
- **Feedforward Neural Network:** Input, hidden, and output layers.  
- **Backpropagation Training:** Implements error calculation and weight adjustment.  
- **Sigmoid Activation Function:** Non-linear transformation of neuron outputs.  
- **Random Weight Initialization:** Initializes weights to small random values.  
- **Solves XOR Problem:** Demonstrates the ability to model non-linear separable datasets.  

---

## Code Explanation  

### 1. **Class Structure**  
The `NeuralNetwork` class encapsulates the entire functionality:  
- **Layers:** `input_layer`, `hidden_layer`, and `output_layer`.  
- **Weights:** `weights_ih` (input to hidden) and `weights_ho` (hidden to output).  
- **Learning Rate:** Set to `0.1` for gradient descent optimization.  

### 2. **Functions in NeuralNetwork**  
- **`sigmoid`:** Applies the sigmoid activation function.  
- **`sigmoid_derivative`:** Computes the derivative of the sigmoid function.  
- **`random_weight`:** Generates a random value between `-1` and `1`.  
- **`feedforward`:** Propagates inputs through the network to produce outputs.  
- **`train`:** Adjusts weights using backpropagation based on the error between predicted and actual outputs.  

### 3. **Main Function**  
- **Initialization:** Creates a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron.  
- **Training:** Trains the network on all combinations of XOR inputs (`{0, 0}`, `{0, 1}`, `{1, 0}`, `{1, 1}`) for 10,000 epochs.  
- **Testing:** Feeds inputs into the trained network to observe outputs.  

---

## How to Run  

### Requirements  
- A C++ compiler (e.g., GCC).  

### Steps to Execute  
1. Copy the code into a file named `xor_nn.cpp`.  
2. Open a terminal/command prompt.  
3. Compile the code:  
   ```bash  
   g++ xor_nn.cpp -o xor_nn -std=c++11  
   ```  
4. Run the executable:  
   ```bash  
   ./xor_nn  
   ```  

---

## Example Output  

The trained neural network produces outputs close to the expected XOR results:  
```  
Training neural network...  

Testing neural network:  
0 XOR 0 = 0.01  
0 XOR 1 = 0.99  
1 XOR 0 = 0.99  
1 XOR 1 = 0.02  
```  

The outputs approximate the correct XOR values:  
- `0 XOR 0 = 0`  
- `0 XOR 1 = 1`  
- `1 XOR 0 = 1`  
- `1 XOR 1 = 0`  

---

## Future Improvements  
- Add support for additional activation functions (ReLU, tanh).  
- Implement momentum or adaptive learning rate to enhance training speed.  
- Extend to multi-class classification problems.  
