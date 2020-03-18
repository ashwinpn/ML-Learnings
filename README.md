## NOTES
## 3-17-2020
- Ian Goodfellow, [Generative Adversarial Networks (GANs) | Artificial Intelligence (AI) Podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0)
1. Deep Learning can be, in simple words, put as taking a thought and refining it again and again, rather than deductive reasoning.
2. Important questions regarding AI - How can we program machines to experience qualitative states of experiences - read as consciousness and self-awareness?
3. Speech recognition is a very interesting and a complex problem, concisely described in the paper ["Hidden Voice Commands"](https://people.eecs.berkeley.edu/~daw/papers/voice-usenix16.pdf). Interestingly, it generated some sounds that a human would NEVER make (see AlphaGo).

- [AlphaGo's develpoment documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y)
1. AlphaGo also played some moves that a human go player would never have been expected to have played = LEARNING.

- NLP
1. Check out wikification.


## Flow GAN's
Combining Maximum Likelihood and Adversarial Learning
[Flow-GAN](https://arxiv.org/abs/1705.08868)

## variational Autoencoders
-[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- Variational Autoencoders (VAEs) are powerful generative models.
- They are one of the most popular approaches to unsupervised learning of complicated distributions.
- VAE's have already shown promise in generating many kinds of complicated data, including handwritten digits, faces, house numbers, CIFAR images, physical models of scenes, segmentation, and predicting the future from static images. 


## Read up on BERT
Bidirectional Encoder Representations from Transformers - NLP Pre-training. 

## xGBoost

Xtreme Gradient Boosting - has given some of the best results recently on problems involving structured data.

## Gradient Boosting
- Why does AdaBoost work so well?
- Gradient Boosting is based on an ensemble based decision tree model, i.e. generating a strong classifier from hypotheses testing of combination of weak classifiers (decision stumps)

## Miscellany
- Keras on Theano optimizers - SAGA, Liblinear (log loss for high dimensional data), ADAM (incremental gradient descent)
- ADAM is basically (RMSprop + momentum turn)
- You can add Nesterov Accelerated Gradient (NAG) to make it better <br>
  [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf) <br>
  [NAG](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/) <br>
- Yet the ADAM optimizer in some cases perfroms poorly as compared to vanilla-SGD?
- Does ReLU always provide a better non-linearity?
  

## Reinforcement Learning
The agent learns from the environment and recives reward/penalties as the result of it's actions. It's objective is to devise policy function in order to maximize cumulative reward.
It's diffrent from supervised and unsupervised learning.
It is based on Markov Decision Processes. But model-free paradigms such as Q-Learning perform better, especially on complex tasks.
- Monte Carlo Policy Gradient (REINFORCE, actor-critic)
- There are problems which arise with gradient values and variance, need to define a baseline and use Bellman's equation
Exploration (exploring new states) v/s Exploitation (maximize overall reward)
- Normal Greedy Approach : Only focus on exploitation
- Epsilon Greedy Approach : Focus on exploration (with probability 1 - epsilon) and exploitation.
- Deep Q Networks (DQN)
  When the number of states / actions become too large, it is more efficient to use Neural Networks. <br>
  In case of DQN, instead of a Bellman Update, we rewrite the Bellman Equation to emulate RMSE form, which woule become our cost function. <be>
- Policy Improvement Methods
- Temporal Difference Methods


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

## Convergence
- Vanilla-SGD achieves 1/t convergence over smoothing of a convex function
- Nesterov Accelerated Gradient (NAG) achieves 1/t.t convergence over smoothing of a convex function
- Newton Methods achieves 1/t.t.t convergence over smoothing of a convex function
- Arora, Mianjy, et.al -- Study convex relaxation based formulations of optimization problems

## Expectation Maximization
- Baum-Welch 
- Forward-Backward Algorithm
