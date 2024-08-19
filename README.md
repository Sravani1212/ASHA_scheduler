Distributed Hyper-Parameter Tuning with Ray Tune and ASHA

# Overview

This project involves experimenting with a distributed hyper-parameter tuning setup using Ray Tune and the Asynchronous Successive Halving Algorithm (ASHA). The goal is to efficiently tune the hyper-parameters of a neural network trained on the CIFAR-10 dataset.

# Features

- Ray Tune Integration: Utilizes Ray Tune for distributed hyper-parameter tuning, enabling parallel experiments across multiple workers.
- ASHA Optimization: Implements the ASHA algorithm for efficient and scalable hyper-parameter optimization.
- CIFAR-10 Dataset: Trains a neural network on the CIFAR-10 dataset, a popular benchmark in image classification.

# Setup and Execution

1. Install Dependencies:
   - Ensure you have Python installed.
   - Install required packages using pip:
     ```sh
     pip install ray[tune] torch torchvision
     ```

2. Run the Tuning Experiment:
   - Execute the script to start the hyper-parameter tuning process:
     ```sh
     python asha_sched.py
     ```

3. Monitor Progress:
   - Ray Tune provides a dashboard for real-time monitoring of the experiments. You can access it by running:
     ```sh
     ray dashboard
     ```

## Conclusion

This project demonstrates the use of Ray Tune and ASHA for distributed hyper-parameter tuning, offering an efficient approach to optimize neural network training on the CIFAR-10 dataset.
