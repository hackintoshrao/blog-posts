# Intuitive and practical guide for building neural networks  from scratch
## Using perceptrons to solve university acceptance problem


### The problem statement 

We’ve got a dataset containing the university admission data. Here’s how the data looks like, [here is the link to the jupyter notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=AZIc9bRMc35B&line=3&uniqifier=1). Just make a copy of the notebook to run it yourself on cloud.

![Image of data](../images/data_sample.png)

The dataset contains 3 columns, the test scores, grades and the result whether student was accepted or not. 1 represents the student being accepted and 0 being rejected. 
We’re gonna solve the classification problem where the test scores and grades are the inputs and the output gonna to be to predict whether the student will be accepted or not.

We’re gonna solve the classification problem where the test scores and grades are the inputs and the output gonna to be to predict whether the student will be accepted or not. Since the output is either of the 2 values, either accepted(represented as 1) or rejected (represented as 0), the problem technically can be termed as a binary classification problem.

Let’s visualize the data, The ones in green represent the students who are accepted the red ones are the those who are rejected. [Here is the link to the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=IUpLCfvEw7kP&line=21&uniqifier=1) .

![Visualizing the data](../images/plot_data.png)
