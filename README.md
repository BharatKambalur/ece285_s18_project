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

   Contains iPython Notebooks that demo different components of our project.
   Also contains notebooks to produce plots for various test and train accuracies.

* Training:

   Contains the iPython Notebooks that can be used to train DenseNet (Baseline, DBPN Pipelined, Bicubic Scaling Pipelined) and RexNeXt (Baseline only)
   
* pymodels:

   Contains the Python files that describe the various pytorch models
   
* models:

   Contains the trained weights for the different networks. (Note: Due to GitHub File size limitation, not all files are uploaded. Please contact the authors to acquire them)
   
* Utils, log:

   Project specific folder that holds Utility scripts and log files respectively
