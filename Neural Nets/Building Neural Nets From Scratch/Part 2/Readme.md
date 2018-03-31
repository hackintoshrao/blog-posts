
# Intuitive and practical guide for building neural networks from scratch

## Part 2: Using perceptrons to solve university acceptance problem

[In the last article](https://medium.com/ai-india/intuitive-and-practical-guide-for-building-neural-networks-from-scratch-d60126645d58) we understood what perceptrons are and how to model a binary classification task using them, the importance of parameters (weights w1, w2 and bias b) and solved the problem of classifying outputs of an AND gate.

 ![](https://cdn-images-1.medium.com/max/1200/1*83ZjwYHc4fbgnJgrcqmX8g.png)

But we manually picked the values for parameters and we left off with the note that we need an automated approach or an algorithm which when fed the data would find out the values for these parameters. Here is the problem statement in short which we set out to solve in the last blog, given the data set of university acceptance based on test score and grades, we need to find the straight line decision boundary which separates the student who were accepted from those who were rejected.

> Thought worm&nbsp;: What if the relationship between inputs and the output is complex and they cannot be separated by a straight line?

* * *

Here are the steps to find the optimal values for parameters w1, w2 and b.

#### Step 1: Start with a assigning a random value for w1, w2 and&nbsp;b

Assign random initial values for w1, w2 and b and see how well it separates the data.

Here is the [notebook cell](https://colab.research.google.com/drive/1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H#scrollTo=Kxi9VVHTjVBH&line=5&uniqifier=1), make a copy and try running it yourself on cloud.

 ![](https://cdn-images-1.medium.com/max/1200/1*aUPbI90BewMLScIGdTxU5g.png)

 ![](https://cdn-images-1.medium.com/max/1200/1*UiqmvavxrkFg56NyHgJijA.png)
***Left:** w1 = 5, w2 = 3, b = -2&nbsp;. **Right:** w1 = 1, w2 = 1, b =&nbsp;-1.4*

The above images shows how well the straight line defined by 2 random set of parameters separate the data.

But which one of the above lines classify the data better&nbsp;? In other words, which one of the above two set of parameters best classify the data?

That brings us to ways of evaluating how good or how bad a given set of parameter classifies the data.

* * *
