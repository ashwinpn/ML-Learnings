## NOTES - IMPORTANT INSIGHTS
## 5-20-2020
- Quantum Computing for Epidemiology - ideal for exponentially growing problems? https://www.youtube.com/watch?v=zOGNoDO7mcU 
- A very interesting ideology at the intersection of Computer Science and Physics
## Research Labs
- Microsoft Station Q , University of California Santa Barbara (UCSB)
- Google Quantum AI Lab
- IBM Quantum Computation Center


## 5-16-2020
- Word alignment and the Expectation Maximization Algorithm. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.5497&rep=rep1&type=pdf

## 4-22-2020
- Sequence to Sequence modelling : How to determine efficiency for (say) Machine Translation tasks / text transduction tasks; How efficient is the BLEU metric?
## 4-6-2020
- https://github.com/ashwinpn/ML-Learnings/blob/master/RL.pdf
- DeepFake detection discussion : https://www.kaggle.com/c/deepfake-detection-challenge/discussion/140236
## 4-5-2020
- Expectation Maximization (Baum - Welch) for Probabilistic Parsing (PCFG, lexicalized PCFG)

## 4-4-2020
- BERT for NLP

## 3-21-2020
- https://towardsdatascience.com/recommendation-system-series-part-2-the-10-categories-of-deep-recommendation-systems-that-189d60287b58
- "It is well-established that neural networks are able to approximate any continuous function with arbitrary precision by varying the activation choices and combinations"
- **https://dl.acm.org/doi/10.1145/2792838.2800187**
- **https://dl.acm.org/doi/10.1145/3240323.3240357**
- https://recsys.acm.org/recsys20/call/#content-tab-1-1-tab
- https://recsys.acm.org/recsys20/challenge/
- https://recsys.acm.org/recsys19/
- https://towardsdatascience.com/recommender-systems-with-deep-learning-architectures-1adf4eb0f7a6
- https://towardsdatascience.com/recommendation-system-series-part-2-the-10-categories-of-deep-recommendation-systems-that-189d60287b58

Basic recap : k-NN, Naive Bayes, SVM, Decision Forests, Data Mining, Clustering, and, Classification

- https://www.slideshare.net/moustaki/time-context-and-causality-in-recommender-systems
- https://www.slideshare.net/linasbaltrunas9/context-aware-recommendations-at-netflix
- https://netflixtechblog.com/to-be-continued-helping-you-find-shows-to-continue-watching-on-7c0d8ee4dab6
- http://news.mit.edu/2017/better-recommendation-algorithm-1206


## 3-20-2020
- [Mckean-Vlasov process](https://en.wikipedia.org/wiki/McKean%E2%80%93Vlasov_process) - in the context of Monte Carlo methods.
1. Monte Carlo methods are ideal for sampling when we have elements which interact with each other - thus its applicability to physics problems.
- A sound advice on visualizations <br> 
https://medium.com/nightingale/ten-considerations-before-you-create-another-chart-about-covid-19-27d3bd691be8
- COVID observations
1. See https://covid19-dash.github.io/
2. Check out https://upload.wikimedia.org/wikipedia/commons/8/86/Average_yearly_temperature_per_country.png
3. Then see https://en.wikipedia.org/wiki/List_of_countries_by_median_age#/media/File:Median_age_by_country,_2016.svg
## 3-19-2020
- The Pitfalls of A/B testing
1. Sequential testing leads to a considerable amount of errors while forming your conclusions - interactions between different elements needs to be taken into account too, while making data driven decisions.
2. The testing should be allowed to run till the end - since we are analysing randomized samples, the test results halfway through and the test results at the end could be polar opposites of each other (!).
3. "The smaller the improvement, the less reliable the results".
4. Need to retest it (at least a couple of times more). Even with a statistically significant result, there’s a quite large probability of false positive error.
- Data Visualization Pitfalls
1. https://junkcharts.typepad.com/
- Bayesian Inference
- Self-Attention in Transformers : One of the things to ponder about, if you want to understand the success of BERT.
- Dealing with outliers and bad data</br>
[QR Factorization and Singular Value Decomposition](https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture09_svd.pdf)
1. Robust regression [https://stats.idre.ucla.edu/r/dae/robust-regression/](https://stats.idre.ucla.edu/r/dae/robust-regression/)
2. Least absolute deviation
3. Iteratively weighted least squares
## 3-18-2020
- [Sentence BLEU score v/s Corpus BLEU score](https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score)
- How vector embeddings of words capture context (GenSim) - "King" - "Man" + "Woman" = "Queen"
- w.r.t calculating P(context-word | center word) = Better alternatives to Singular Vector Decomposition (SVD)? 
 1. [SVD tutorial](https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm)
 2. [Krylov-Schur approach to the truncated SVD](http://www.cs.ox.ac.uk/files/721/NA-08-03.pdf) 
- Transfer Learning considerations
1. What factors influence **When and how to fine-tune?** - size of the dataset, similarities with the original dataset.
2. Pre-trained network weights provided : https://github.com/BVLC/caffe/wiki/Model-Zoo

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
- ADAM is basically (RMSprop + momentum term)
- You can add Nesterov Accelerated Gradient (NAG) to make it better <br>
  [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf) <br>
  [NAG](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/) <br>
- Yet the ADAM optimizer in some cases performs poorly as compared to vanilla-SGD?
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
a significantly large, labeled dataset for tackling the problem successfully.

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
