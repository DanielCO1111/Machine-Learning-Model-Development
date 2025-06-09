# Machine-Learning-Model-Development
This project demonstrates the implementation and application of foundational machine learning algorithms using Python, Numpy, Pandas, and Matplotlib. The notebooks include hands-on solutions for real-world classification and clustering tasks such as university admissions prediction and digit recognition.

## Logistic Regression

The Logistic Regression part divided into two main parts:
1. *Part 1: Neural Network using NumPy*  
2. *Part 2: Neural Network in PyTorch* 

The goal is to implement and analyze neural network models first from scratch in NumPy, then using PyTorch.

### Key Features
- Comprehensive Data Analysis: Detailed exploration and manipulation of datasets to uncover underlying patterns and relationships.
- Predictive Modeling: Introduction to building and refining predictive models using advanced statistical methods.
- Interactive Visualizations: Leveraging Python's visualization libraries to create interactive plots that effectively communicate the results of data analysis.

### Data Cleaning and Preparation
- Techniques for handling missing data, anomalies, and extracting relevant features for analysis.
![image](https://github.com/user-attachments/assets/26df7643-258f-4bc9-9164-d4d90f3b61bf)

### Statistical Analysis and Data Visualization
- Application of statistical tests to determine significant differences and correlations within the data. 
- Advanced visualization techniques to create dynamic plots and charts for an impactful presentation of data insights.

![image](https://github.com/user-attachments/assets/7d6479bf-974a-41b8-9142-1ae4d845b3ca)

### Predictive Analytics and Case Study
- Construction of models to forecast future trends based on historical data.
- A real-world case study to apply learned techniques, demonstrating the workflow from raw data to actionable insights.
![image](https://github.com/user-attachments/assets/781cbcaa-4e18-4217-8a4d-891cf8f9a2e1)

---

## Neural Network and KNN

The Neural Network and KNN part divided into two main parts:

1. *Feedforward Neural Network using NumPy:*
   
- Implement and analyze a simple neural network for classification from scratch using NumPy.

2. *K-Nearest Neighbors (KNN) Algorithm:*
   
- Apply and evaluate the KNN algorithm for supervised classification, including parameter selection and model assessment.

The goal is to gain hands-on experience with two classic classification algorithms—neural networks (implemented from the ground up) and KNN—comparing their effectiveness on different datasets.

### Key Features

1. Custom Model Implementation: Build neural networks and KNN classifiers from first principles.

2. Hyperparameter Tuning: Experiment with neural network architecture and KNN's 'K' value to optimize accuracy.

3. Comprehensive Evaluation: Visualize decision boundaries and model predictions using Matplotlib.

### Data Preparation
Includes steps for data normalization, feature selection, and handling of training/test splits to ensure fair algorithm comparison.

#### Dataset
For part 1:

The MNIST (Modified National Institute of Standards and Technology database) dataset contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits (10 digits). The handwritten digit images have been size-normalized and centered in a fixed size of 28x28 pixels.

![image](https://github.com/user-attachments/assets/3b4eb621-f9de-4b39-b242-1a9665e63de0)

For part 2:

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes - 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'.

![image](https://github.com/user-attachments/assets/b98e12f5-8d49-44e8-9c41-b8777b5862dc)



### Performance Visualization & Analysis
Plot of three random images and print their labels:

![image](https://github.com/user-attachments/assets/96d32f63-a5a9-4a2f-baf5-ad5edac24ab0)


After choosing a random image and passing it through the network. It should return a prediction - confidences for each class. The class with the highest confidence is the prediction of the model for that image:

![image](https://github.com/user-attachments/assets/e038c939-7b27-4fd2-abc0-740cb5e081da)

![image](https://github.com/user-attachments/assets/2d5dc74c-5f70-4528-860d-a80b03c00318)


### Neural Network - Training

1. Split the dataset into a training set and a validation set. Train-set size: 80% of the total data. Val-set size: the rest 20%.
   
2. Create a dataloader for each set (train_loader and val_loader, see Section 2 for examples).

3.Choose hyperparameters (for now we choose learning_rate=0.005 and num_epochs=5). 

4. Use SGD (Stochastic Gradient Descent) as the optimizer.
   
5. Since it is a multi-class classification task, use "negative log-likelihood loss" as the loss criterion.

6. Train your model on the train-set and evaluate it on the validation-set.

7. During training, for each epoch, track the training loss and validation loss.

training loss drop with each epoch:

![image](https://github.com/user-attachments/assets/fc84b83f-ee72-4c75-9401-3e306c7251d5)

Plot train loss and validation loss as a function of epoch:

![image](https://github.com/user-attachments/assets/fae06544-08b4-43c0-a305-d9fa5517257a)

With the network trained, we can check out it's predictions:

![image](https://github.com/user-attachments/assets/3dc6d7a2-0171-4634-9dd9-a78e19a24e59)

The model's accuracy on the validation-set:

![image](https://github.com/user-attachments/assets/f3667e37-c93e-4ff6-9b94-dabd47baf10c)

---

## KMeans Clustering
This notebook focuses on unsupervised clustering using the K-means algorithm:

### Main Parts:

1. K-Means Clustering from Scratch with NumPy

2. Step-by-step implementation of K-means, from centroid initialization to convergence.

3. Cluster Analysis and Visualization

4. Assign data points to clusters and interpret the results visually.

*The goal is to explore how K-means can be used to identify natural groupings in data, with all components built manually in NumPy.*

### Key Features
Full Clustering Pipeline: Develop all clustering steps (init, assignment, update) without ML libraries.

In-Depth Cluster Evaluation: Assess clustering effectiveness with inertia and visual inspection.

Result Interpretation: Use plots to communicate clustering outcomes and data structure.

### Data Preprocessing
Covers standardization and dimensionality reduction (if needed) to improve clustering quality.

### Visualization and Results
Includes scatter plots and color-coded cluster assignments for clear interpretation.

---

### Requirements
- Python 3.7 or higher  
- Jupyter Notebook or JupyterLab  
- NumPy  
- Matplotlib  
- PyTorch  
- Torchvision  

You can install the dependencies using pip:
```bash
pip install numpy matplotlib torch torchvision
