## NOTES - IMPORTANT INSIGHTS
## 6-6-2020
## 6-5-2020
## 6-4-2020
## 6-2-2020
- THE CURIOUS CASE OF NEURAL TEXT DeGENERATION : https://arxiv.org/pdf/1904.09751.pdf
- Beam Search Strategies for Neural Machine Translation : https://www.aclweb.org/anthology/W17-3207.pdf
## 5-31-2020
- An overview of Bayesian analysis - http://lethalletham.com/Letham_bayesian_analysis.pdf 
- Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization - https://arxiv.org/abs/2001.11659
- Forecasting at scale (Time series) - Facebook Prophet : https://peerj.com/preprints/3190.pdf
## 5-29-2020
- Approximation Schemes for ReLU Regression : https://arxiv.org/pdf/2005.12844.pdf
- Distributed Algorithms for Covering, Packing and
Maximum Weighted Matching : https://arxiv.org/pdf/2005.13628.pdf
## 5-27-2020
## 5-25-2020
- The PGM-index: a fully-dynamic compressed learned index
with provable worst-case bounds : https://dl.acm.org/doi/pdf/10.14778/3389133.3389135
## 5-24-2020
- NLP and Knowledge graphs - generate word embeddings from knowledge graphs.
- While training with Adam helps in getting fast convergence, the resulting model will often have worse generalization performance than when training with SGD with momentum. Another issue is that even though Adam has adaptive learning rates its performance improves when using a good learning rate schedule. Especially early in the training, it is beneficial to use a lower learning rate to avoid divergence. This is because in the beginning, the model weights are random, and thus the resulting gradients are not very reliable. A learning rate that is too large might result in the model taking too large steps and not settling in on any decent weights. When the model overcomes these initial stability issues the learning rate can be increased to speed up convergence. This process is called learning rate warm-up, and one version of it is described in the paper Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
- What is perplexity? What is its place in NLP?
```
Perplexity is a way to express a degree of confusion a model has in predicting. More entropy = more confusion. Perplexity is used to evaluate language models in NLP. A good language model assigns a higher probability to the right prediction.
```
- What is the problem with ReLu?

```
1] Exploding gradient(Solved by gradient clipping)
2] Dying ReLu : No learning if the activation is 0 (Solved by parametric relu)
3] Mean and variance of activations is not 0 and 1. (Partially solved by subtracting around 0.5 from activation. Better explained in fastai videos)
```

## 5-23-2020
- Programming Tensor cores for CUDA - https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
- Bias Variance Decompositions using XGBoost - https://devblogs.nvidia.com/bias-variance-decompositions-using-xgboost/
- Run XGBoost : Decreasing Test Error.
 
 ```
import csv
import numpy as np
import os.path
import pandas
import time
import xgboost as xgb
import sys
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
dmatrix_train_filename = "higgs_train.dmatrix"
dmatrix_test_filename = "higgs_test.dmatrix"
csv_filename = "HIGGS.csv.gz"
train_rows = 10500000
test_rows = 500000
num_round = 1000

plot = True

# return xgboost dmatrix
def load_higgs():
    if os.path.isfile(dmatrix_train_filename) 
      and os.path.isfile(dmatrix_test_filename):           
        dtrain = xgb.DMatrix(dmatrix_train_filename)
        dtest = xgb.DMatrix(dmatrix_test_filename)
        if dtrain.num_row() == train_rows and dtest.num_row() == test_rows:
            print("Loading cached dmatrix...")
            return dtrain, dtest

    if not os.path.isfile(csv_filename):
        print("Downloading higgs file...")
        urlretrieve(data_url, csv_filename)

    df_higgs_train = pandas.read_csv(csv_filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)
    dtrain = xgb.DMatrix(df_higgs_train.ix[:, 1:29], df_higgs_train[0])
    dtrain.save_binary(dmatrix_train_filename)
    df_higgs_test = pandas.read_csv(csv_filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)
    dtest = xgb.DMatrix(df_higgs_test.ix[:, 1:29], df_higgs_test[0])
    dtest.save_binary(dmatrix_test_filename)

    return dtrain, dtest


dtrain, dtest = load_higgs()
param = {}
param['objective'] = 'binary:logitraw'
param['eval_metric'] = 'error'
param['tree_method'] = 'gpu_hist'
param['silent'] = 1

print("Training with GPU ...")
tmp = time.time()
gpu_res = {}
xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
          evals_result=gpu_res)
gpu_time = time.time() - tmp
print("GPU Training Time: %s seconds" % (str(gpu_time)))

print("Training with CPU ...")
param['tree_method'] = 'hist'
tmp = time.time()
cpu_res = {}
xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
          evals_result=cpu_res)
cpu_time = time.time() - tmp
print("CPU Training Time: %s seconds" % (str(cpu_time)))

if plot:
    import matplotlib.pyplot as plt
    min_error = min(min(gpu_res["test"][param['eval_metric']]), 
                    min(cpu_res["test"][param['eval_metric']]))
    gpu_iteration_time = 
        [x / (num_round * 1.0) * gpu_time for x in range(0, num_round)]
    cpu_iteration_time = 
        [x / (num_round * 1.0) * cpu_time for x in range(0, num_round)]
    plt.plot(gpu_iteration_time, gpu_res['test'][param['eval_metric']], 
             label='Tesla P100')
    plt.plot(cpu_iteration_time, cpu_res['test'][param['eval_metric']], 
             label='2x Haswell E5-2698 v3 (32 cores)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Test error')
    plt.axhline(y=min_error, color='r', linestyle='dashed')
    plt.margins(x=0)
    plt.ylim((0.23, 0.35))
    plt.show()
 ```
 
## 5-22-2020
- Variational Inference for NLP - http://nlp.cs.berkeley.edu/tutorials/variational-tutorial-slides.pdf
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
