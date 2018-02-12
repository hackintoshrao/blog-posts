# Intuitive and practical guide for building neural networks  from scratch
## Using perceptrons to solve university acceptance problem

Here is the list of concepts which are covered in this blog,
- Perceptrons.
- Modelling the binary classification task using perceptron.
- Building AND gate using perceptrons.

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

Let’s learn some fundamentals which are necessary to solve the problem and then get back to crack it.
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
score = w1 * x1 + w2 * x2 + b 

Output is: 
     1 if score > 0
     0 if score < 0
```

In case of our university admission data an output of 1 corresponds to the student being accepted and 0 being rejected.

The goal is to learn the values w1, w2 and b in such a way that the perceptron network output (which is either 0 or 1) matches with the ground truth label y for most the training examples. We have 100 student records at our disposal for solving the university admission problem, we at-least expect that the output of perceptron matches with ground truth label for 90+ cases.

Once the values of w1, w2 and b are learned in a way which satisfies the goal the following equation represents the line which can best separate 2 classes of data.

```
w1 * x1 + w2 * x2 + b = 0
```
This fits the general equation of straight line which is represented by

```
A * x + B * y + C = 0
```

[Check this link](https://www.mathsisfun.com/algebra/line-equation-general-form.html) to know more about equation of a straight line.

Let’s consider an example, The following equation represents the straight line which separates the 2 classes of data, the accepted ones from the rejected ones for the university admission problem,

<em><strong>2x1 + x2–18 = 0</em></strong> , where w1 = 2, w2 = 1 and c = -18.

![line example](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/line_example.png?raw=true)

Now you know the importance of having right set of values for w1, w2 and b. Because these parameters characterize the straight line.

Hence, finding the right set of parameters for the perceptron network which could best classify our dataset inturn gives us a straight decision boundary which best separates the two classes of data.

Still finding it hard to get the intuition? Well, don’t worry, let’s solve very simple problem which using perceptrons and in the which let’s gain more intuition about how it works and how it can be used to draw a straight line boundary to separate 2 classes of data.

---

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

Let’s plot these 4 training examples and see how it looks, [Here is the link of the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=gqJPfMff3io5), again, make a copy and run it yourself.

![And gate plot and code](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_plot_code.png?raw=true)

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

![AND gate perceptron](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_perceptron.png?raw=true)

> Refresher: That’s what a linear classifier for a binary classification task does, it separates the inputs whose outputs are  1 from those whose outputs are 0 by drawing a decision boundary using a straight line.

---

Let’s start with random values for <em><strong>w1</em></strong>, <em><strong>w2</em></strong> and <em><strong>b</em></strong> to begin with and see how well it separates the 2 classes of data.

Set <em><strong>w1 = 3, w2 = 2, b = -1</em></strong> to begin with, lets replace these values in the perceptron model and try couple of things,

First, check if the line represented by these parameters correctly separates data.

```
# the decision boundary is represented by 
w1 * x1 + w2 * x2 + b = 0
# Replace w1, w2 and b with the random values we chose.
# Here is our straight line equation.
3 * x1 + 2 * x2 - 1 = 0
```

![Straight line classifying AND gate dataset](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_classify_1.png?raw=true)

Clearly the straight line <em><strong>3 * x1 + 2 * x2 – 1 = 0</em></strong> which is represented by parameters <em><strong>w1 = 3, w2 = 2 and b = -1</em></strong> is not able to separate the red ones from green point.

[Here is the link to the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=BbnMb09s3mZf&line=31&uniqifier=1), just make a copy and run on your own. Try the code with different values for w1, w2 and b.

Let’s check the perceptrons output for these parameters and compare it with the ground truth label.

```python

m collections import OrderedDict
import numpy as np


# Setting the values for parameters w1, w2 and b.
# Try the code with various parameter values.
w1 = 3
w2 = 2
b =  -1


def find_output(score):
    """
    The perceptron output is 
      1 if score >= 0
      0 if score < 0
    """
    if score >= 0:
        return 1
    
    return 0

def find_score(x1, x2, w1=w1, w2=w2, b=b):
    """
    The perceptron score is calculated by 
    score = x1 * w1 + x2 * w2 + b
    """
    
    score = x1 * w1 + x2 * w2 + b
    return score 


def find_perceptron_prediction(x1,x2, w1=w1,w2=w2, b=b):
    """
    1. Find the score 
    2. Find the perceptron prediction.
    
    """
    score = find_score(x1,x2)
    prediction = find_output(score)
    return prediction
  

and_gate_input_0 = np.array([1, 1, 0, 0])
and_gate_input_1 = np.array([0, 1, 0, 1])
# The output is 1 only in case where both the corresponding inputs are 1.
and_gate_output  = np.array([0, 1, 0, 0])

# Find the perceptron output for the AND gate dataset using the following parameters, 
# w1 = 3, w2 = 2, b = -1.

prediction = []

# for all 4 samples in the AND dataset find the perceptron prediction.
for i in range(4):
    pred = find_perceptron_prediction(and_gate_input_0[i], and_gate_input_1[i]) 
    prediction.append(pred)
    
df= pd.DataFrame(OrderedDict( ( ('Input 0', pd.Series(and_gate_input_0)), ('Input 1', pd.Series(and_gate_input_1)),
                               ('Actual AND output', pd.Series(and_gate_output)), ('perceptron_prediction', pd.Series(prediction)))))

print(df)


```

![Compating AND output and the perceptron output](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_pred_out_1.png?raw=true)

As you can clearly see in the table above the actual AND gate output and the perceptron prediction doesn’t match. [Here](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=E0wyrHZjR0nx&line=4&uniqifier=1) is the link to notebook cell, make a copy of it and run it on your own, try various values of w1, w2 and b.

With parameters set to values <em><strong>w1 = 1, w2 = 1 and b = -1.5</em></strong> the perceptron model is able to separate the 2 classes of data and predict the output correctly too. [Here is the link to notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=E0wyrHZjR0nx&line=4&uniqifier=1)

![AND correct classify](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_good.png?raw=true)

With parameters set to values w1 = 1, w2 = 1 and b = -1.5 the perceptron prediction matches the actual AND gate output, Here is the [link to the notebook cell](https://colab.research.google.com/notebook#fileId=1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H&scrollTo=XsjbTr_AmS9k&line=50&uniqifier=1),
![AND gate data classified correctly](https://github.com/hackintoshrao/blog-posts/blob/master/Neural%20Nets/Building%20Neural%20Nets%20From%20Scratch/Part%201/images/and_correct_table.png?raw=true)

---

As an exercise try to set the parameters w1, w2 and b such that it satisfies the OR gate output.

But what about the initial problem we set out to solve? To draw the straight line to correctly classify or separate the accepted and rejected students?


In case of classifying AND gate output we set the parameters w1, w2 and b manually by hand. This brute force approach doesn’t scale when we have hundreds, thousands or sometimes millions of data points. And here we had only 2 inputs, how would we do it in case there’s a tens or even hundreds of output?

In the next post of the series we’ll learn about an algorithm which would learn the parameters (weights and bias) of the perceptron network on their own. We’ll have some fun learning some fundamental calculus too and then solve the university admission classification problem in an automated way.

Note: A lot of emphasis has been laid on making this blog really easy to be followed even for an absolute beginner. If you have questions or you wanna drop in a feedback on improving the blog please feel free to comment. If you find this useful don’t forget to clap and share. See you soon in the next blog. Till then, happy learning!

---

Additional sources :

    [More on bias in neural nets](https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks)
    [Role of activation function neural nets](https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network-How-does-this-function-in-a-human-neural-network-system)
    [More on perceptrons](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html)






