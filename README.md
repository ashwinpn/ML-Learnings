## xGBoost

Xtreme Gradient Boosting - has given some of the best results recently on problems involving structured data.

## Miscellany
- Keras on Theano optimizers - SAGA, Liblinear (log loss for high dimensional data), ADAM (incremental gradient descent)
- ADAM is basically (RMSprop + momentum turn)
- You can add Nesterov Accelerated Gradient (NAG) to make it better

## Reinforcement Learning
The agent learns from the environment and recives reward/penalties as the result of it's actions. It's objective is to devise policy function in order to maximize cumulative reward.
It's diffrent from supervised and unsupervised learning.

## Transfer Learning
Use a model trained on one problem to do predictive modelling on another problem.
For instance, say you have a image classification task.
You can use the VGG Model shell, conveniently provided by Oxford at their Vector Graphics Group Website.
You definitely would need to change the last few layers based on your task, and other changes would require
hypotheses testing / domain knowledge.

Transfer learning really improves efficiency in the case where we need to perform supervised learning tasks, and we require
a significantly large, labelled dataset for tackling the problem successfully.
