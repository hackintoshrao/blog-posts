# Intuitive and practical guide for building neural networks  from scratch
## Using perceptrons to solve university acceptance problem


### The problem statement 

We’ve got a dataset containing the university admission data. Here’s how the data looks like, [here is the link to the jupyter notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=AZIc9bRMc35B&line=3&uniqifier=1). Just make a copy of the notebook to run it yourself on cloud.

![Image of data](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/data_sample.png?raw=true)

The dataset contains 3 columns, the test scores, grades and the result whether student was accepted or not. 1 represents the student being accepted and 0 being rejected. 
We’re gonna solve the classification problem where the test scores and grades are the inputs and the output gonna to be to predict whether the student will be accepted or not.

We’re gonna solve the classification problem where the test scores and grades are the inputs and the output gonna to be to predict whether the student will be accepted or not. Since the output is either of the 2 values, either accepted(represented as 1) or rejected (represented as 0), the problem technically can be termed as a binary classification problem.

Let’s visualize the data, The ones in green represent the students who are accepted the red ones are the those who are rejected. [Here is the link to the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=IUpLCfvEw7kP&line=21&uniqifier=1) .

![Visualizing the data](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/plot_data.png?raw=true)

Let’s start with a simple classification problem. We need a way to separate the red ones from the green ones. The nature of the data from the plot shows us that the these can be almost separated just by using straight line. So here’s the problem definition. Need to figure out this straight which best separates the students who are accepted (in green), from those who are rejected (in red). This line would also help us make future decision of whether to accept a student or not based on their test score and grades.Once the answer is found out it would look something like this,

![The straight line solution](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/plot_good.png?raw=true)

> We assume that there’s a functional mapping between the set of inputs and outputs, and during training we try to make sense of it and try to find that relationship.

---

## Perceptrons
Perceptrons are the building blocks of neural networks, they are the basic units from which more complex neural networks are built. They can be used as simple linear classifiers, which can be used to draw a straight line to separate the linearly separable data points and that’s precisely what we want.

> The perceptron is a mathematical model of a biological neuron.

Perceptron and neural networks?! It smells like some biological inspiration, isn’t it?. Yes, you are right and here’s why,
![Perceptron Biological connection](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/perceptron_neuron.png?raw=true)

- While in actual neurons the dendrite receives electrical signals from the axons of other neurons, in the perceptron these electrical signals are represented as numerical values.
- At the synapses between the dendrite and axons, electrical signals are modulated in various amounts. This is also modeled in the perceptron by multiplying each input value by a value called the weight.
- An actual neuron fires an output signal only when the total strength of the input signals exceed a certain threshold. We model this phenomenon in a perceptron by calculating the weighted sum of the inputs to represent the total strength of the input signals, and applying a step function on the sum to determine its output.
- As in biological neural networks, this output is fed to other perceptrons.

---

>The output of the perceptron is linear equation of inputs and weights. The important fact to notice is that its a linear equation.


![Perceptron output](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/perceptron_output.png?raw=true)

- The values x1, x2… xn are the inputs to the perceptrons, these are represented as nodes of the perceptron network.
- The values w1, w2, …wn on the edges are called the weights of the network. These weights are like the knobs which controls the output value of the network.
- The weight w1 is associaated with input x1, w2 with x2 and so on…
- The last node doesn’t correspond to any input features, it has a constant value of 1,  the edge contains a value b called the bias of the network. Don’t worry much about the importance of bias, we’ll have better understanding about weights and bias as we progress further. For now consider it as a number which giving much heed to its importance.

> The number of weights is equal to number of inputs. And the output of the perceptron is either 0 or 1.

- The output of the network is calculated by first calculating the product of the inputs with their corresponding weights and arriving at a score.

```
score = w1 * x1 + w2 * x2 +  ..... + wn * xn + 1 * b
the output is 
1  if the score > 0
0  if the score < 0
```

> Hold on for a moment !!!! How does perceptron help us find the straight line which help us separate the accepted and rejected candidates in case of the university admission problem?

That’s a valid question! Let’s figure that out so that it makes sense to continue to learn about perceptrons in order to crack our problem.

---

## Modelling the problem using perceptrons

In the university admission problem are setup out to solve there are 2 inputs, the <em><strong><test scores (x1)</em></strong> and <em><strong> grades (x2) </em></strong>, we saw that the perceptron score is a linear function of inputs and weights, let <em><strong>w1 </em></strong> and <em><strong>w2 </em></strong> be the weights that for inputs <em><strong>x1 </em></strong> and <em><strong>x2</em></strong>, let a real number<em><strong>b</em></strong>  be the bias of the network. 

Now according to the perceptron model the output score of the network with inputs <em><strong>x1</em></strong> and <em><strong>x2</em></strong>, and with weights <em><strong>w1 </em></strong> and <em><strong>w2 </em></strong> is 
```
w1 * x1 + w2 * x2 + b 
```
remember <em><strong>x1 </em></strong> and <em><strong>x2</em></strong> are our inputs test scores and grades, so the above equation is linear, and is similar to the equation of a straight line, which is, 
```
a * x + b * y + c = 0
```

[Check this link](https://www.mathsisfun.com/algebra/line-equation-general-form.html) to know more about equation of a straight line.

Hence the perceptron model for 2 inputs represents a straight line in <em><strong>x</em></strong> and <em><strong>y</em></strong> axis. Hold on! Isn’t that what we need??!! A straight line to separate the accepted students from the rejected ones. Therefore we can use the perceptron model and find numerical values for <em><strong>w1</em></strong>, <em><strong>w2 </em></strong> and <em><strong>b</em></strong> in such a way that it would represent a line which could best separate the classes in our university admission dataset.

Still finding it hard to get the intuition? Well, don’t worry, let’s solve very simple problem which using perceptrons and in the which let’s gain more intuition about how it works and how it can be used to draw a straight line boundary to separate 2 classes of data.

Let’s start with the simpler task of classifying the outputs of a 2 input  AND gate using perceptrons,

Here’s how the input-output relationship looks like for an AND gate with 2 inputs. As you see the inputs are discrete and they are either 0 or 1. We need to find out the line which separates the input (1,1) with output 1 from rest of the inputs whose output is 0.

![And gate table](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_table.png?raw=true)


Let’s analyse the above training data first,

```
 1. We have 4 training examples.
 2. Each training example has 2 input features, input_1 and input2.
 3. Each training example has just 1 output.
 4. The output's are either 0 or 1, so it's a binary classification task.
```

Here’s how we would model the linear classification task for the above analyzed dataset using the perceptron model,

Let’s plot these 4 training examples and see how it looks, [Here is the link of the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=gqJPfMff3io5), again, make a copy and run it yourself.

![And gate plot and code](../images/and_plot_code.png)

As you can see, the plot (1, 1) is in green, corresponding to output value 1, and the other 3 input plots are in red indicating that their output is 0.

Here’s how we would model the linear classification task for the above analyzed dataset using the perceptron model,

```
1. Choose a number for weights w1, w2 and bias b in a way that it satisfies the conditions mentioned in the points to follow.
 
2. The output of the perceptron 
     is 1 if w1 * input_1 + w2 * input_2 + b > 0
     is 0 if w1 * input_1 + w2 * input_2 + b < 0

3. For the AND gate dataset we know the perceptron should output value of 1 only in case where both the inputs are 1. 

4. Which means for inputs input_1 = 1 and input_2 = 1, the output of the perceptron w1 * input_1 + w2 * input_2 + b should be > 0, and for rest of the inputs the output of the perceptron should be < 0.

```
The image below depicts the kind of straight line we are expecting, a line which can  separate the input (1,1) from the rest.

![AND gate perceptron](../images/and_perceptron.png)

> Refresher: That’s what a linear classifier for a binary classification task does, it separates the inputs whose outputs are  1 from those whose outputs are 0 by drawing a decision boundary using a straight line.


