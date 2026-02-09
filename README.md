# EECS498: Deep Learning for Computer Vision

Course materials and implementations from the University of Michigan's Deep Learning for Computer Vision course.

- **Course Website:** https://web.eecs.umich.edu/~justincj/teaching/eecs498/
- **Instructor:** [Justin Johnson](https://web.eecs.umich.edu/~justincj/)

## About

This repository contains my solutions and implementations for the EECS498 Deep Learning for Computer Vision course assignments and projects.

## Assignment Overview

| S/No | Area of Focus | Notebooks | Source Code |
|------|---------------|-----------|-------------|
| 1 | **PyTorch Fundamentals & k-NN** | [pytorch101.ipynb](A1/pytorch101.ipynb), <br>[knn.ipynb](A1/knn.ipynb) | [pytorch101.py](A1/pytorch101.py), <br>[knn.py](A1/knn.py) |
| 2 | **Linear Classifiers & Neural Networks** | [linear_classifier.ipynb](A2/linear_classifier.ipynb), <br>[two_layer_net.ipynb](A2/two_layer_net.ipynb) | [linear_classifier.py](A2/linear_classifier.py), <br>[two_layer_net.py](A2/two_layer_net.py) |
| 3 | **Fully-Connected Networks & CNNs** | [fully_connected_networks.ipynb](A3/fully_connected_networks.ipynb), <br>[convolutional_networks.ipynb](A3/convolutional_networks.ipynb) | [fully_connected_networks.py](A3/fully_connected_networks.py), <br>[convolutional_networks.py](A3/convolutional_networks.py) |
| 4 | **Object Detection: 1-stage (FCOS) & 2-stage (Faster R-CNN)** | [one_stage_detector.ipynb](A4/one_stage_detector.ipynb), <br>[two_stage_detector.ipynb](A4/two_stage_detector-no-outputs.ipynb)__*__ | [one_stage_detector.py](A4/one_stage_detector.py), <br>[two_stage_detector.py](A4/two_stage_detector.py) |
| 5 | **Image Captioning (RNNs) & Transformers** | [rnn_lstm_captioning.ipynb](A5/rnn_lstm_captioning.ipynb), <br>[transformers.ipynb](A5/transformers.ipynb) | [rnn_lstm_captioning.py](A5/rnn_lstm_captioning.py), <br>[transformers.py](A5/transformers.py) |
| 6 | **Generative Models (VAEs & GANs), Feature Visualization & Style Transfer** | [variational_autoencoders.ipynb](A6/variational_autoencoders.ipynb), <br>[generative_adversarial_networks.ipynb](A6/generative_adversarial_networks.ipynb), <br>[network_visualization.ipynb](A6/network_visualization.ipynb), <br>[style_transfer.ipynb](A6/style_transfer.ipynb) | [vae.py](A6/vae.py), <br>[gan.py](A6/gan.py), <br>[network_visualization.py](A6/network_visualization.py), <br>[style_transfer.py](A6/style_transfer.py) |


## Note on Large Files

Due to GitHub's file size limitations, some Jupyter notebooks with outputs that exceed 25MB are denoted in the table above with (__*__). For these files:
- Cleaned versions (without outputs) are provided in this repository 

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# EECS 498-007 / 598-005: Deep Learning for Computer Vision

## Assignment 1: PyTorch 101 and k-Nearest Neighbor
* **PyTorch 101:** Learn the basics of working with tensors in PyTorch. Implement fundamental operations like tensor creation, indexing, and reshaping.
* **k-Nearest Neighbor (kNN):** Implement a k-Nearest Neighbor classifier. Practice the train/val/test split workflow and hyperparameter tuning using validation data on the CIFAR-10 dataset.

---

## Assignment 2: Linear Classifiers
* **Linear Classifiers (SVM & Softmax):** Implement a Multiclass Support Vector Machine (SVM) and a Softmax classifier from scratch. Focus on vectorized gradient computation and numeric gradient checking.
* **Two-Layer Neural Network:** Build a modular two-layer neural network classifier.
* **MNIST Challenge:** Manually set weights for a Two-Layer Network to solve a simple digit recognition task.

---

## Assignment 3: Fully-Connected and Convolutional Networks
* **Fully-Connected Neural Networks:** Implement modular backpropagation and various update rules (SGD+Momentum, RMSProp, Adam). Includes Batch Normalization and Dropout implementation.
* **Convolutional Neural Networks (CNN):** Implement the forward and backward passes for Convolutional and Max-Pooling layers. Train a deep CNN on CIFAR-10.

---

## Assignment 4: Object Detection
* **One-Stage Detector:** Implement **FCOS** (Fully-Convolutional One-Stage Object Detector), an anchor-free design. Train and evaluate on the PASCAL VOC 2007 dataset.
* **Two-Stage Detector:** Implement a detector similar to **Faster R-CNN**, combining a Region Proposal Network (RPN) with a second-stage recognition network.

---

## Assignment 5: Sequence Modeling and Attention
* **Image Captioning with RNNs & LSTMs:** Implement vanilla Recurrent Neural Networks and LSTMs to generate captions for images.
* **Spatial Attention:** Augment the captioning model with a spatial attention mechanism over image regions.
* **Transformers:** Implement the building blocks of the Transformer architecture (Multi-head Attention) and test it on a synthetic arithmetic dataset.

---

## Assignment 6: Generative Models & Visualization
* **Variational Autoencoders (VAE):** Implement VAEs and Conditional VAEs on the MNIST dataset for image generation and latent space interpolation.
* **Generative Adversarial Networks (GAN):** Implement fully-connected and convolutional GANs (DCGAN) to generate realistic hand-written digits.
* **Network Visualization:** Use image gradients to create Saliency Maps, Adversarial Examples, and Class Visualizations.
* **Style Transfer:** Create artistic images by combining the content of one image with the style of another using feature reconstruction loss.

