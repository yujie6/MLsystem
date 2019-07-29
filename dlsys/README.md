# About
testcase for [PPCA 2019](https://acm.sjtu.edu.cn/wiki/PPCA_2019) machine learning system

# Test Description
| no |    name     | test item |
|----| ----------- | --- |
| 1  | adder       | basic computation graph |
| 2  | initializer | global initializer |
| 3  | assign      | assign op |
| 4  | context     | 'with' statement support |
| 5  | mnist_grad    | automatic differentiation |
| 6  | GD optimizer   | gradient descent optimizer |
| 7  | multilayer perceptron | relu activation |
| 8  | Adam optimizer       | adam optimizer |
| 9  | CNN Check       | check the correctness of CNN |
| 10 | CNN GD       | simple CNN |
| 11 | CNN Adam with Dropout | simple CNN with dropout |

# How to run test
```bash
python3 run_test.py <name_of_your_model>
```

Since our API is the same as tensorflow, you can use tensorflow to pass all the tests
```bash
python3 run_test.py tensorflow
```