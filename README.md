# ECE 285: Image Processing for Machine Learning

Description
===========
This repository contains all the code used for our project "Improving Classification with a Pipelined Architecture using Super-Resolution Methods." The code was developed by Team Perceptive Perceptrons composed of Bharat Kambular, Erik Seetao, Joseph Mattern, and Sharla Chang.


Requirements
============
The following Python Packages are required: numpy, matplotlib, Pillow, pytorch (tested with 0.3.1)

Install package the packages as follow :
$ pip install --user <package_name>


Code organization
=================
* Demos:

... Contains iPython Notebokks that demo different components of our project.
... Also contains notebooks to produce plots for various test and train accuracies.

* Training:

--- Contains the iPython Notebooks that can be used to train DenseNet (Baseline, DBPN Pipelined, Bicubic Scaling Pipelined) and RexNeXt (Baseline only)
