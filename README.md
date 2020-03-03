## xGBoost

Xtreme Gradient Boosting - has given some of the best results recently on problems involving structured data.

## Miscellany
- Keras on Theano optimizers - SAGA, Liblinear (log loss for high dimensional data), ADAM (incremental gradient descent)
- ADAM is basically (RMSprop + momentum turn)
- You can add Nesterov Accelerated Gradient (NAG) to make it better <br>
  [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf) <br>
  [NAG] (https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/) <br>
  

## Reinforcement Learning
The agent learns from the environment and recives reward/penalties as the result of it's actions. It's objective is to devise policy function in order to maximize cumulative reward.
It's diffrent from supervised and unsupervised learning.
It is based on Markov Decision Processes. But you can prefer model-free paradigms such as Q-Learning.
- Monte Carlo Policy Gradient (REINFORCE, actor-critic)
- There are problems which arise with gradient values and variance, need to define a baseline and use Bellman's equation
Exploration (exploring new states) v/s Exploitation (maximize overall reward)
- Deep Q Networks (DQN)


## Transfer Learning
Use a model trained on one problem to do predictive modelling on another problem.
For instance, say you have a image classification task.
You can use the VGG Model shell, conveniently provided by Oxford at their Vector Graphics Group Website.
You definitely would need to change the last few layers based on your task, and other changes would require
hypotheses testing / domain knowledge.

Transfer learning really improves efficiency in the case where we need to perform supervised learning tasks, and we require
a significantly large, labelled dataset for tackling the problem successfully.

## Visualization
- Matplotlib is still popular in general
- Can also use Pandas for visualization
- Plotly.JS, D3.JS  for beautiful outputs that could be rendered in Browsers
- Bokeh is becoming popular of late; It has bindings in Python, Lua, Julia, Java, Scala.

## Regularization Techniques
Regularization is used for reducing overfitting.

- L1, L2 regularization : regularization over weights
- ElasticNet - L1 + L2 regularization
- Adversarial Learning - Problems faced : Some tasks which can be very easily performed by humans have been found to be very difficult for a computer. For example, if you introduce a little noise to the photo of a Lion , it may not be recognized as a Lion (or worse, not as an animal at all). Thus, you voluntarily introduce noise to the extended dataset to improve efficiency. This is called jittering.
- Dropout - Eradicate some neural network nodes / layers to improve performance.
- Tikhonov regularization / Ridge Regression - Regularization of [ill posed](https://en.wikipedia.org/wiki/Well-posed_problem) problems 

## Probabilistic Graphical Models
- Inferential Learning
- Markov Random Fields
- Conditional Random Fields
- Bayesian Networks

## Stochastic Gradient Descent
- What is the ideal batch size?
- Dealing with Vanishing Gradients (very small values of d/dw)

## CNN's
- Pooling + Strides is used for downsampling of the feature map.
- AlexNet, GoogLeNet, VGG, DenseNet.
