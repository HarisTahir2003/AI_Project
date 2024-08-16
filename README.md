# Hi, I'm Haris! 👋

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 

# Artificial Intelligence Project

The two Jupyter Notebooks in this repository explore two major Machine Learning algorithms (K-Nearest Neighbours and Regression Trees), with a particular focus on accurately predicting the motion of micro-robots in a complex environment with onstacles. The notebooks are structured to provide a comprehensive understanding of these algorithms, and include practical implementations, visualizations, and model evaluations. <br> 

The AI_Project folder contains the following files:
- A KNN.ipynb file (Jupyter Notebook) that contains all the code regarding the KNN part of the assignment including text blocks explaining portions of the code
- A corresponding KNN.py file
- A RegressionTree.ipynb file (Jupyter Notebook) that contains all the code regarding the Regression Tree part of the assignment including text blocks explaining portions of the code
- A corresponding RegressionTree.py file
- three .png files that are screenshots of the plots in the KNN Jupyter Notebook
- two .png files that are screenshots of the plots in the Regression Tree Jupyter Notebook
- a 1200-second video recording `(training_data.mp4)` of the robot's movement within the wooden box environment.
- a text file  `(training_data.txt)`containing the robot's coordinates
- a test video `(test01.mp4)` 
- a test txt file `(test01.txt)` 

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Project Structure](#project-structure)
4. [Data](#data)
5. [Training and Evaluation](#training-and-visualization)
6. [Lessons](#lessons)
7. [Screenshots](#screenshots)
   
## Introduction

K-Nearest Neighbors (KNN) is a simple, non-parametric classification and regression algorithm. It works by finding the k closest training examples to a given test point and making predictions based on these neighbors. For classification, KNN assigns the class most common among the neighbors, while for regression, it averages the values of the neighbors. <br>

A regression tree is a type of decision tree used for predicting continuous outcomes. It splits the data into subsets based on feature values, aiming to minimize variance within each subset. The process continues recursively, creating a tree-like structure where each node represents a decision based on a feature, and each leaf node represents a predicted value. Regression trees are useful for capturing non-linear relationships and interactions between features, but they can be prone to overfitting if not properly pruned or regularized. <br>

 This assignment provides a clear and concise example of how to implement the KNN and Regression Tree algorithms from scratch using Python.
 
## Installation Requirements

To run the both the notebooks, you will need the following packages:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these packages using pip:

```bash
 pip install numpy
```
```bash
 pip install pandas
```
```bash
 pip install matplotlib 
```
```bash
 pip install scikit-learn
```
After installing the libraries, simply run the 'Imports' code block to enable their usage in the file.

Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Project Structure

The first Jupyter Notebook (KNN.ipynb) is organized into the following sections:
<ul>
<li> Problem Description: Overview of what the objective of the project is about</li> 
<li> Time Series and Lookback: An introduction and explanation to the concepts of time-series and lookback in the field of Artificial Intelligence </li>
<li> Dataset Overview: a description of what the training and testing data contains </li>
   
<li> Part 1A: KNN from Scratch <br>
&emsp; 1) Imports: libraries imported to implement this part <br>
&emsp; 2) Data Loading and Preprocessing: Steps to load and preprocess the dataset <br>
&emsp; 3) Model Training: Training the KNN model from scratch <br>
&emsp; 4) Model Evaluation: Evaluating and analyzing the performance of the model, using a plot and a written explanation </li> 
&emsp; 5) Visualization of Actual and Predicted Path: a visual comparison of the actual trajectory of the micro-robot and the one predicted by the algorithm </li> <br> 
<li> Part 1B: KNN using scikit-learn </li> 
&emsp; Implementation of the KNN algorithm using the scikit-learn library
  <br>
</ul>

The second Jupyter Notebook (RegressionTree.ipynb) is organized into the following sections:

<li> Part 2: Regression Tree <br>
&emsp; 1) Imports: libraries imported to implement this part <br>
&emsp; 2) Regression Tree Implementation: loading the data and training the Regression Tree model using the DecisionTreeRegressor() function of scikit-learn library <br>
&emsp; 3) Model Evaluation: Evaluating and analyzing the performance of the model, using a plot and a written explanation </li> 
&emsp; 4) Visualization of Actual and Predicted Path: a visual comparison of the actual trajectory of the micro-robot and the one predicted by the algorithm </li> <br> 


## Data

Training Data
  - A 1200-second video recording `(training_data.mp4)` of the robot's movement within the wooden box environment. This video is captured at 30 frames per second (fps).
  - A text file  `(training_data.txt)`containing the robot's coordinates, with 30 values recorded for each second (since video is 30 fps).

* Testing Data
  - A test video `(test01.mp4)`, 60 seconds long recorded at 30 fps.
  - A test txt file `(test01.txt)` following the same format as the `training_data.txt` file.


## Training and Visualization

The entire training process alongside the maths involved is explained in detail in the jupyter notebook. 

## Lessons

An AI project such as the one implemented here, involved many challenges, including:

1. **Handling Time Series Data:**
   - **Challenge:** Working with time series data requires careful consideration of the temporal order and dependencies between observations. This can be tricky when predicting future values based on past data.
   - **Solution:** To manage this, I implemented a lookback mechanism, which involved using previous observations to predict future values. For both KNN and Regression Tree models, this allowed me to capture temporal dependencies effectively.

2. **Implementing KNN from Scratch:**
   - **Challenge:** Building the KNN algorithm from scratch without relying on libraries like scikit-learn involved creating functions for distance calculation, finding nearest neighbors, and handling ties in predictions.
   - **Solution:** I wrote custom functions for Euclidean distance and nearest neighbor selection. To handle ties, I implemented a mechanism to decrement k until a clear prediction was obtained, ensuring robust and accurate results.

3. **Evaluating Model Performance:**
   - **Challenge:** Choosing the right value of k for the KNN model and the lookback size for the Regression Tree model required extensive evaluation. The performance needed to be assessed using metrics like RMSE.
   - **Solution:** I plotted RMSE values against different k values and lookback sizes to identify the optimal parameters. This involved iterating through various values and analyzing trends to select the best-performing configurations.

4. **Handling Model Complexity and Overfitting:**
   - **Challenge:** With increasing model complexity (e.g., higher k values or longer lookback periods), there was a risk of overfitting, where the model might perform well on training data but poorly on unseen data.
   - **Solution:** I monitored performance metrics across various configurations and chose parameters that balanced model complexity and generalization. For KNN, I observed the trend in RMSE with varying k values, and for Regression Trees, I tested different lookback sizes to find the optimal trade-off.


## Screenshots
<h3> K-Nearest Neighbour (KNN) </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for increasing values of k. Further explanation of the results of the plot are explained in detail in the Jupyter Notebook. </h4>
<img src="pic11.png" width="450px"> <br> 

<h4> 2. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm <strong> implemented from scratch </strong>. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times.  </h4>
<img src="pic12.png" width="450px"> <br> 

<h4> 3. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm <strong> implemented using the scikit-learn library </strong>. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times.  </h4>
<img src="pic13.png" width="450px"> <br> 

<h3> Regression Tree </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for increasing values of Lookback Size. Further explanation of the results of the plot are explained in detail in the Jupyter Notebook. </h4>
<img src="pic21.png" width="450px"> <br> 

<h4> 2. This image shows the trajectory of the actual path of the micro-robot along with the trajectory predicted by the KNN algorithm implemented using the scikit-learn library. You can change the value of 'start_second' in the code to compare the two trajectories for different six-second sets of times. </h4>
<img src="pic22.png" width="450px"> <br> 

 
## License

[MIT](https://choosealicense.com/licenses/mit/)

