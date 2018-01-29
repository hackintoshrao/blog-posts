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

Let’s start with a simple classification problem. We need a way to separate the red ones from the green ones. The nature of the data from the plot shows us that the these can be almost separated just by using straight line. So here’s the problem definition. Need to figure out this straight which best separates the students who are accepted (in green), from those who are rejected (in red). This line would also help us make future decision of whether to accept a student or not based on their test score and grades.Once the answer is found out it would look something like this,

![The straight line solution](../images/plot_good.png)

---

### The straight line 
Now we know that we are trying find this straight line which best separates the data. The equation of a straight looks like this ,

<em><strong> y = m * x + c</em></strong>

<em><strong> m * x + c – y = 0 </em></strong>

In the above equation m is the slope of the line and c is the y intercept of the line, both these values together characterize the line. Our goal is to find the values for <em><strong>m</em></strong> and <em><strong>c</em></strong> which will represent a line which will best separate the admitted students from the rejected ones. [Here is an useful link](https://www.mathsisfun.com/equation_of_line.html) if you want to know about the equation of the straight line and intuition behind it.

The process in which we use various techniques to find the best values for these parameters is called <em><strong>learning</em></strong> or <em><strong>training</em></strong>.We’ll be using the 100 student records available, they’ll be used to train the parameters, these are collectively called as <em><strong>training data</em></strong> and each student record is referred to as a <em><strong>training example</em></strong>. In machine learning terminology the values <em><strong>m</em></strong> and <em><strong>c</em></strong> are called the parameters and are represented by <em><strong>w</em></strong> and <em><strong>b</em></strong>.

So now the equation becomes,

<em><strong>w * x + b – y = 0</em></strong>

<em><strong>w * test_score + b - grades= 0</em></strong>

Let’s understand the necessary fundamental concepts and solve a relatively simpler problem before we tackle the current one.

>We assume that there’s a functional mapping between the set of inputs and outputs, and during training we
> try to make sense of it and try to find that relationship.

---

## Perceptrons
Perceptrons are the building blocks of neural networks, they are the basic units from which more complex neural networks are built. They can be used as simple linear classifiers, which can be used to draw a straight line to separate the linearly separable data points and that’s precisely what we want.

> The perceptron is a mathematical model of a biological neuron.

Perceptron and neural networks?! It smells like some biological inspiration, isn’t it?. Yes, you are right and here’s why,
![Perceptron Biological connection](../images/perceptron_neuron.png)

- While in actual neurons the dendrite receives electrical signals from the axons of other neurons, in the perceptron these electrical signals are represented as numerical values.
- At the synapses between the dendrite and axons, electrical signals are modulated in various amounts. This is also modeled in the perceptron by multiplying each input value by a value called the weight.
- An actual neuron fires an output signal only when the total strength of the input signals exceed a certain threshold. We model this phenomenon in a perceptron by calculating the weighted sum of the inputs to represent the total strength of the input signals, and applying a step function on the sum to determine its output.
- As in biological neural networks, this output is fed to other perceptrons.
![Perceptron output](../images/perceptron_output.png)


