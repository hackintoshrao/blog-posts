
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

#### Step 2: Calculate the error or&nbsp;loss
​
Think about this? What could be the possible ways to evaluate how well or how bad a set of parameters classifies the data?&nbsp;
​
One way is to find the number of correctly classified or classified data points. Let’s see the number of misclassifed or wrongly classified data points for above two set of parameters&nbsp;.&nbsp;
​
The points in blue represents the points which are wrongly classified by the straight line.
​
 ![](https://cdn-images-1.medium.com/max/1200/1*yKqMc58lJ7D2HHD85qpD6g.png)
​
 ![](https://cdn-images-1.medium.com/max/1200/1*KUrDQtvckhaZyco6x15Xag.png)
​
So the number of wrongly classified points by parameters (1, 1, -1.4) is 24, and parameters (5, 3, -2) is 39. So definitely the first set of parameters are better. [Here is the link to the notebook cell](https://medium.com/r/?url=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H%23scrollTo%3D98OoWUp6FcxQ%26line%3D104%26uniqifier%3D1), try running it yourself by changing the values of w1, w2 and b.&nbsp;
​
Though the number of wrongly classified points helps us compare the effectiveness of different sets of parameters, its not sufficient to optimize the values of w1, w2 and b. We need a continuous value error function which captures more information about how badly the straight line is classifying the points.&nbsp;
​
We have 100 student records for the university admission data, how about we use a metric which is a function of the ground truth label and the prediction.&nbsp;
​
Consider the ground truth label to be y and the prediction to be y^ (spelled has y hat), the goal here is to reduce the difference between y and y^ for all the examples, that is to choose values for parameters w1, w2 and b which reduces `(y - y^)`&nbsp;. How about we choose the popular squared error function?&nbsp;
​
```
error = (1/2) * (y - y^)²
```
​
But the squared error function doesn’t work well for classification tasks. In the next section we’ll see an illustration on how does it fails to enrapture the degree of correctness of the classification.
​
Here is the algorithm to find the cumulative error using the squared error function.
​
```
cumulative_error = 0 initialize w1, w2, and b.
```
​
```
For i range(total_records): 1. find prediction y^ 2. find the error e = (1/2) * (y - y^) 3. Sum up the errors, cumulative_error = cumulative_error + e
```
​
```
# Find the average for the cumulative errorcumulative_error = cumulative_error / total_num_records
```
​
The cumulative error is called the loss of the network, the loss function is represented by `L(y, y^)`&nbsp;.
​
Let’s take the 3 sets of parameters we considered in the above examples and calculate the loss and compare them, the better set of parameters should have the least loss.
​
 ![](https://cdn-images-1.medium.com/max/800/1*ttVD3MyzsR7agkvlXeAISw.png)
​
 ![](https://cdn-images-1.medium.com/max/800/1*6NixXy2WjdxuOaRBvH5uJA.png)
​
 ![](https://cdn-images-1.medium.com/max/800/1*YVkN5WUZ6bo8Txgbgu2qRw.png)
*Plots for parameters with their number of wrongly classified points and the&nbsp;loss*
​
 ![](https://cdn-images-1.medium.com/max/1600/1*-zicsD5zUuDSBKbZBLoLmQ.png)
​
I know what you are thinking now, that the squared error loss seems to reduce with the reduction in number of wrongly classified points and hence we can use it as the error function. Isn’t it? Right guess, but it doesn’t quite work like that when we peek into it carefully. Let’s again generate the above table, but this time with constant values of w2 and b, we’ll generate bunch of values for w1 and obtain the number of wrongly classified points and the loss value.&nbsp;
​
If the squared error function were to rightly capture the correctness of the classification it the loss should reduce with reduction in number of wrongly classified points. Let’s see what happens.
​
 ![](https://cdn-images-1.medium.com/max/1200/1*SExiJS_m8GOwmJ9AN45piw.png)
​
 ![](https://cdn-images-1.medium.com/max/1200/1*4DV-ofZRk0v2ZM0sxllK_g.png)
*The number of classified points with its loss for various values of&nbsp;weight1*
​
 ![](https://cdn-images-1.medium.com/max/1200/1*U3ziMAMS7hDqOjdk0bZaAA.png)
​
Take a look at those marked points circle in red, the value of loss is at its least value of 0.095604 when the number of wrongly classified points is 20.
​
But when the number of wrongly classified points are 10 the value of loss is expected to reduce compared to what it was when it misclassified 20 points, but instead the loss increases gradually to 0.22&nbsp;.&nbsp;
​
This is like an intuitive and empirical proof that the squared error function doesn’t actually encapture the goodness of a classifier, it basically tries to fit a regression line and fails to understand that the outputs 1 and 0 are the values which just indicates the category of data and these are not continuous valued outputs. In the next blog we’ll discuss more about the loss functions and optimization techniques that can be used in classifiers.&nbsp;
​
> But how do we know optimize the parameters to fit a straight line which can best classify these two classes of&nbsp;data?
​
* * *


 ![](https://cdn-images-1.medium.com/max/1200/1*4DV-ofZRk0v2ZM0sxllK_g.png)
*The number of classified points with its loss for various values of&nbsp;weight1*

 ![](https://cdn-images-1.medium.com/max/1200/1*U3ziMAMS7hDqOjdk0bZaAA.png)

Take a look at those marked points circle in red, the value of loss is at its least value of 0.095604 when the number of wrongly classified points is 20.

But when the number of wrongly classified points are 10 the value of loss is expected to reduce compared to what it was when it misclassified 20 points, but instead the loss increases gradually to 0.22&nbsp;.&nbsp;

This is like an intuitive and empirical proof that the squared error function doesn’t actually encapture the goodness of a classifier, it basically tries to fit a regression line and fails to understand that the outputs 1 and 0 are the values which just indicates the category of data and these are not continuous valued outputs. In the next blog we’ll discuss more about the loss functions and optimization techniques that can be used in classifiers.&nbsp;

> But how do we know optimize the parameters to fit a straight line which can best classify these two classes of&nbsp;data?

* * *

#### The optimization technique

Phew, finally we’ve reached the climax. The issue with the squared error function was that it took the large error values into consideration even from the correctly classified points. We need to extract as much information as we can only from the wrongly classified points.

&nbsp;There are two possible cases where the straight line boundary would be wrong,&nbsp;

**Case 1:** When the straight line is too much into the red points area.

 ![](https://cdn-images-1.medium.com/max/1200/1*jLYy-pntqzeCkfbWqKt2EQ.png)

- This is the case where lot of red point are misclassified and almost none of the green points are wrongly classified.
- The parameters to the straight line are 5, 3, -2.
- The prediction of perceptron in this case for all the misclassified red points is 1, but their actual labels are 0.&nbsp;
- In this case the line has to be pulled back towards the green points cluster.
- This can be done by subtracting the values of these wrongly classified points to the parameters.

Here is the sequence of steps to be followed for this case,&nbsp;

```
initialize w1, w2 and b
```

```
For all wrongly classified red points(that is when y = 0 and y^ = 1)
```

```
# x1 represents the x axis value of misclassified red point # x1 represents the test score in the university admission data w1 = w1 - (a small number) * x1 # x2 represents the y-axis value of misclassified red point # x2 represents the grades. w2 = w2 - (a small number) * x2
```

```
# update bias b = b - (a small number)
```

To make sure that the update to parameters w1, w2 and b are not very large, we multiply the values with a small number before the update. This is small number is called **_learning rate_** and is represented by value **α** (alpha). Usually the learning rate will of order 0.1 to 0.0001. The convergence of the parameters to optimal values slows down with decrease in learning rate. But the large value of learning rate would cause the parameters to overshoot from the optimal value.

Here is the snippet for updating parameter for the first case,&nbsp;

Let us run one iteration of the optimization and see whether the classifier gets better. The number of wrongly classified points should reduce if the optimization is effective. Here are the results before and after the classification,&nbsp;

 ![](https://cdn-images-1.medium.com/max/1200/1*9ZZUGI0Qq1x8Qa3rgwLnQw.png)

The first plot corresponds to perceptrons prediction before the optimization, it wrongly classifies 39 data points with its initial set of parameters&nbsp;

But after the optimization the performance significantly improves and the number of wrongly classified points reduces to just 6.

Here is the[link to the notebook](https://medium.com/r/?url=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1_u0KMavhqmyTsLCIce0ay7J9Aao-vE-H%23scrollTo%3DlW_eY7WYAv2W%26line%3D152%26uniqifier%3D1), make a copy, running it yourself and play around. Try and run with various values of learning rate. With decrease in learning rate the optimization on the parameters has to be run multiple times.

Below are series of images which shows the goodness of the classification after one run of optimization for various values of learning rates.

 ![](https://cdn-images-1.medium.com/max/800/1*sg7FY-1yagPyDF0qQ2BwSQ.png)

 ![](https://cdn-images-1.medium.com/max/800/1*DU1KmY3ebtEjfLP-kU9nnQ.png)

 ![](https://cdn-images-1.medium.com/max/800/1*EHgwop58qJ4a6j3d73PnDw.png)

 ![](https://cdn-images-1.medium.com/max/800/1*livFZPw87jMNedlLPBVdGA.png)

 ![](https://cdn-images-1.medium.com/max/800/1*eYkIoG2Or6EsSNdESw9leQ.png)

 ![](https://cdn-images-1.medium.com/max/800/1*RzNx_fn292Cbq1iTGqYI7g.png)

Don’t assume that the optimization becomes less effective with smaller learning rates, the optimization just gets slower and has to be run multiple times for parameters to reach the optimal value. Too small a value for le
