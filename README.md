```
import numpy as np
import tensorflow as tf
from tabulate import tabulate
```

## Activation Functions
Neural Network example:

*<img src="https://drive.google.com/uc?id=1eaFYKdTDJhQsnY291Leo_rsnIDTvBG0m" width="400" height="200" />*


$𝑧_1 = (𝑥_1 ∗ 𝑤_1) + (𝑥_2 ∗ 𝑤_2) + 𝑏_1$  
$𝑧_1$ =(0.1∗0.15)+(0.2∗0.05)+0.33 
$𝑧_1$ = 0.355

For Signmoid activation function, output values:  
$𝑎_1 =𝑓(𝑧_1)= {1\over 1+ 𝑒^{−(0.355)}}$ = 0.588 

$𝑧_2 = (𝑎_1 ∗ 𝑤_2) + 𝑏_2$ 

$𝑧_2 = (0.588 ∗ 0.36) + 0.56$   
$𝑧_2$ = 0.772 

$𝑎_2 =𝑓(𝑧_2)= {1\over 1+ 𝑒^{−(0.772)}}$ = 0.648 


```
# Helper function to calculate the node value of hidden layer 1
# Hidden layer1 consists of only one node
def output_hidden_layer_1(a_1):
  b_1 = tf.constant(0.33)
  w_2 = tf.constant(0.36)
  b_2 = tf.constant(0.56)
  z_2 = a_1*w_2 + b_2
  return z_2

# Helper function for a simple ReLu function
def relu(input):
  if input > 0:
	  return input
  else:
	  return 0
```


```
# Neural net calculation
z_1 = tf.constant(0.355)

# Linear activation function
a_1_lin = z_1
z_2_lin = output_hidden_layer_1(a_1_lin) # recalculate z_2 from a_1_lin output
a_2_lin = z_2_lin

# Tanh activation function: (𝑒^𝑥 − 𝑒^(−𝑥)) / (𝑒^𝑥 + 𝑒^(−𝑥))
a_1_tanh = tf.tanh(z_1)
z_2_tanh = output_hidden_layer_1(a_1_tanh)
a_2_tanh = tf.tanh(z_2_lin)

# ReLu
# In this case, Relu output will be the same as linear, since the outputs are
# not linear, but to stay in the spirit of things, selection will be used
a_1_relu = relu(z_1)
z_2_relu = output_hidden_layer_1(a_1_relu)
a_2_relu = relu(z_2_relu)

# Call the .numpy() method to post only the value of the tensor in eager mode
print("Sigmoid function: ", "a_1 = ", 0.588, "a_2 = ", 0.648)
print("Linear function: ", "a_1 = ", a_1_lin.numpy(), "a_2 = ", a_2_lin.numpy())
print("Tanh function: ", "a_1 = ", a_1_tanh.numpy(), "a_2 = ", a_2_tanh.numpy())
print("ReLU function: ", "a_1 = ", a_1_relu.numpy(), "a_2 = ", a_2_relu.numpy())
```

    Sigmoid function:  a_1 =  0.588 a_2 =  0.648
    Linear function:  a_1 =  0.355 a_2 =  0.6878
    Tanh function:  a_1 =  0.34080228 a_2 =  0.59656686
    ReLU function:  a_1 =  0.355 a_2 =  0.6878


### Gradients of Activation Functions
<img src="https://drive.google.com/uc?id=1PIn0Gk3Dru9VzA3dj72ND9gCgIbxyFCO" width="400" height="200" />

ReLu is designed to help with the vanishing gradient problem.
These gradients are derived with the chain rule.





```
# Gradient helper functions
# Takes in tensorflow object as input
def sigmoid_gradient(x):
  grad = tf.sigmoid(x) * (1 - tf.sigmoid(x));
  return grad

def tanh_gradient(x):
  grad = 1 - tf.square((tf.tanh(x)))
  return grad

def relu_gradient(x):
    output_list = []
    for val in x.numpy():
      if val > 0:
        output_list.append(1)
      else:
        output_list.append(0)
    output_tensor = tf.constant(output_list)
    return output_tensor
  
```


```
# Input list
# Make sure to set dtype to float, since default dtype for lists can give you
# Integer vs float errors (-4 is interpreted as integer type, while 0.5 float)
inputs = tf.constant([-4,0.5,4], dtype = "float32")

# Outputs
sigmoid_grad_outputs = sigmoid_gradient(inputs)
tanh_grad_outputs = tanh_gradient(inputs)
relu_grad_outputs = relu_gradient(inputs)

# Print gradients at each point
print("Sigmoid Gradient:")
print("At x = -4: " , "Sigma prime = ", sigmoid_grad_outputs.numpy()[0])
print("At x = 0.5: " , "Sigma prime = ", sigmoid_grad_outputs.numpy()[1])
print("At x = 4: " , "Sigma prime = ", sigmoid_grad_outputs.numpy()[2])
print("Tanh Gradient:")
print("At x = -4: " , "Tanh prime = ", tanh_grad_outputs.numpy()[0])
print("At x = 0.5: " , "Tanh prime = ", tanh_grad_outputs.numpy()[1])
print("At x = 4: " , "Tanh prime = ", tanh_grad_outputs.numpy()[2])
print("ReLu Gradient:")
print("At x = -4: " , "ReLu prime = ", relu_grad_outputs.numpy()[0])
print("At x = 0.5: " , "ReLu prime = ", relu_grad_outputs.numpy()[1])
print("At x = 4: " , "ReLu prime = ", relu_grad_outputs.numpy()[2])
```

    Sigmoid Gradient:
    At x = -4:  Sigma prime =  0.017662734
    At x = 0.5:  Sigma prime =  0.23500372
    At x = 4:  Sigma prime =  0.017662734
    Tanh Gradient:
    At x = -4:  Tanh prime =  0.0013411045
    At x = 0.5:  Tanh prime =  0.7864477
    At x = 4:  Tanh prime =  0.0013411045
    ReLu Gradient:
    At x = -4:  ReLu prime =  0
    At x = 0.5:  ReLu prime =  1
    At x = 4:  ReLu prime =  1


### Softmax equation:
Usually used only in the output layer of classification models.
$$S(y_i) = {e^{y(i)}\over \sum_j e^{y_j}}$$



```
# Two random input logits
V1 = np.array([2.3, 1.2, 0.3, 0.0])
V2 = np.array([1.9, 1.7, 2.6, 0.2, 1.3])

# Calculate output from input logits
V1_softmax = tf.nn.softmax(V1)
V2_softmax = tf.nn.softmax(V2)
```

## Binary Cross-Entropy / Log Loss

𝐶𝑜𝑠𝑡 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 = −( (𝑡𝑎𝑟𝑔𝑒𝑡 ∗ log(𝑐𝑜𝑚𝑝𝑉𝑎𝑙𝑢𝑒) + (1 − 𝑡𝑎𝑟𝑔𝑒𝑡) ∗ log(1 − 𝑐𝑜𝑚𝑝𝑉𝑎𝑙𝑢𝑒)) )


```
computed_value= tf.constant([0.95,0.8,0.6,0.4,0.1])

# Case for binary target of 1 or 0
target = 0;
cost_function_target_0 = -1 *( target*tf.math.log(computed_value) + (1-target) * 
                         tf.math.log(1 - computed_value) )
target = 1;
cost_function_target_1 = -1 *( target*tf.math.log(computed_value) + (1-target) * 
                         tf.math.log(1 - computed_value) )

# Output results to terminal
print("When Target = 0: ")
print("Computed Value: ", computed_value.numpy()[0], " Cost Function: ", cost_function_target_0.numpy()[0])
print("Computed Value: ", computed_value.numpy()[1], " Cost Function: ", cost_function_target_0.numpy()[1])
print("Computed Value: ", computed_value.numpy()[2], " Cost Function: ", cost_function_target_0.numpy()[2])
print("Computed Value: ", computed_value.numpy()[3], " Cost Function: ", cost_function_target_0.numpy()[3])
print("Computed Value: ", computed_value.numpy()[4], " Cost Function: ", cost_function_target_0.numpy()[4])
print("When Target = 1: ")
print("Computed Value: ", computed_value.numpy()[0], " Cost Function: ", cost_function_target_1.numpy()[0])
print("Computed Value: ", computed_value.numpy()[1], " Cost Function: ", cost_function_target_1.numpy()[1])
print("Computed Value: ", computed_value.numpy()[2], " Cost Function: ", cost_function_target_1.numpy()[2])
print("Computed Value: ", computed_value.numpy()[3], " Cost Function: ", cost_function_target_1.numpy()[3])
print("Computed Value: ", computed_value.numpy()[4], " Cost Function: ", cost_function_target_1.numpy()[4])
```

    When Target = 0: 
    Computed Value:  0.95  Cost Function:  2.995732
    Computed Value:  0.8  Cost Function:  1.609438
    Computed Value:  0.6  Cost Function:  0.9162908
    Computed Value:  0.4  Cost Function:  0.5108256
    Computed Value:  0.1  Cost Function:  0.105360545
    When Target = 1: 
    Computed Value:  0.95  Cost Function:  0.051293306
    Computed Value:  0.8  Cost Function:  0.22314353
    Computed Value:  0.6  Cost Function:  0.5108256
    Computed Value:  0.4  Cost Function:  0.9162907
    Computed Value:  0.1  Cost Function:  2.3025851


## Argmax Function

<img src="https://drive.google.com/uc?id=18f3-KiMgUU_Mez3HWWWSGRbcT0SVeKpq" width="600" height="200" />


Image taken from: Physics Dept, Cornell University


```
# Random tensor
a = tf.constant([[5,2,3],[26,56,92],[3,0,26]])

# Axis=0 tells you the maximum value location in each column (reduces along the rows)
a1 = tf.argmax(a,axis=0)
# Axis=1 tells you the maximum value location in each row (reduces along the columns)
a2 = tf.argmax(a,axis=1)

# Output argmax tensors to output window
print(a1)
print(a2)
```

    tf.Tensor([1 1 1], shape=(3,), dtype=int64)
    tf.Tensor([0 2 2], shape=(3,), dtype=int64)


## XOR Gate Neural Network
Input data and correct output:

<img src="https://drive.google.com/uc?id=1dgqtiFQPmhIpVdrX498kiIVAgAz-e3T-" width="300" height="150" />

Neural Network Model:

<img src="https://drive.google.com/uc?id=1Q57rgI33Krm5hememSpJPNcDKGKz-ZT1" width="500" height="300" />


```
# Calculate output based on input from XOR gate
def output_hidden_layer1(input):
  # Given weights
  w_1= tf.constant([[-4, -6, -5],[3, 6, 4]], dtype = 'float32')
  b_1 = tf.constant([-2, 3, -2], dtype = 'float32')
  
  # Multiply input of 1x2 tensor by 2x3 tensor and add 1x3 bias
  output_H1 = tf.matmul(input, w_1) + b_1
  output_H1_activation = tf.sigmoid(output_H1)

  # Hidden layer two
  # w2 Must be a column vector
  w_2 = tf.constant([[5],[-9],[7]], dtype = 'float32')
  b_output = tf.constant([[4]], dtype = 'float32') # single bias from 1 node output
  output_layer_pre_activation = tf.matmul(output_H1_activation, w_2) + b_output
  output_layer_activation = tf.sigmoid(output_layer_pre_activation)
  
  return output_layer_activation

```


```
# Surround with double brackets to make the sizes match with weights 
# All inputs for XOR gate
input1 = tf.constant([[0,0]], dtype = 'float32')
input2 = tf.constant([[1,0]], dtype = 'float32')
input3 = tf.constant([[0,1]], dtype = 'float32')
input4 = tf.constant([[1,1]], dtype = 'float32')

comp_output_1 = output_hidden_layer1(input1)
comp_output_2 = output_hidden_layer1(input2)
comp_output_3 = output_hidden_layer1(input3)
comp_output_4 = output_hidden_layer1(input4)

# 𝐸𝑟𝑟𝑜𝑟 = (𝐶𝑜𝑚𝑝𝑢𝑡𝑒𝑑𝑂𝑢𝑡𝑝𝑢𝑡 − 𝑇𝑟𝑢𝑒𝑂𝑢𝑡𝑝𝑢𝑡)^2
error1 = (comp_output_1 - tf.constant([[0]], dtype = 'float32'))
error2 = (comp_output_2 - tf.constant([[1]], dtype = 'float32'))
error3 = (comp_output_3 - tf.constant([[1]], dtype = 'float32'))
error4 = (comp_output_4 - tf.constant([[0]], dtype = 'float32'))

#Output to terminal
l = [[0, 0, 0, comp_output_1, error1], [1, 0, 1, comp_output_2, error2],
     [0, 1, 1, comp_output_3, error3], [1,1,0, comp_output_4, error4]]
table = tabulate(l, headers=['Input 1', 'Input 2', 'True Output', 'Computed Output', 'Error'], tablefmt='orgtbl')

print(table)
```

    |   Input 1 |   Input 2 |   True Output |   Computed Output |       Error |
    |-----------+-----------+---------------+-------------------+-------------|
    |         0 |         0 |             0 |         0.0413786 |  0.0413786  |
    |         1 |         0 |             1 |         0.973193  | -0.0268073  |
    |         0 |         1 |             1 |         0.992013  | -0.00798655 |
    |         1 |         1 |             0 |         0.0179147 |  0.0179147  |

