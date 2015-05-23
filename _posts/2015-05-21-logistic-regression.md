---
layout: post
title: "Logistic Regression"
description: A small introduction to logistic regression classification algorithm.
categories:
- machine_learning
---

A small introduction to logistic regression classification algorithm.

___

Logistic regression is a discriminative classification model.

##Types of classification models

1.  **Generative classification models**: This approach models the class conditional densities
    $$p(\textbf{x}|C_k)$$ and class priors $$p(C_k)$$. And knowing these two quantities,
    we can get the posterior class probabilities, using [Bayes theorem][bayes wikipedia link]:
    
    $$
    \begin{equation} \label{eq}
        p(C_k|\textbf{x})=\frac{p(\textbf{x}|C_k)p(C_k)}{p(\textbf{x})}
    \end{equation}
    $$
    
    This model has more parameters and will spend more time on training with high dimensional data.
    The marginal density $$p(\textbf{x})$$ can be used to detect new data points that has low probability and, therefore, our classification won't be precise in that case. by $\ref{eq}$

2.  **Discriminative models**: This models find $$p(C_k|\textbf{x})$$ probabilities directly.
    In this case make decisions based only on the posterior class probabilities. Probabilites
    can help for example in case when we have a point that has nearly equal probabilities for
    each class. In that case we can avoid making decisions because they will be not so accurate.
    We won't be able to do it when using discriminant functions that will be described in the next step.
    This model is also easier to train than generative model.

3.  **Discriminant function approach**: Approach that is only has a discriminative function
    $$f(\textbf{x})$$ that maps each input to some particular class $$C_k$$. This approach
    doesn't have a probabilistic interpretation and usually faster to train. You gain speed but loose
    probabilistic interpretation that can be useful in some cases.

##Deriving the logistic regression equation

First, let's consider the example of two classes. The posterior probability of class $$C_1$$
can be written as:

$$p(C_1|\textbf{x}) = \frac{p(\textbf{x}|C_1)p(C_1)}{p(\textbf{x}|C_1)p(C_1) + p(\textbf{x}|C_2)p(C_2)}$$

Then, we will use a different presentation of the same equation:


$$p(C_1|\textbf{x}) = \sigma(a(\textbf{x}))$$

Where:

$$
\begin{equation} \label{logistic}
    \sigma(a) = \frac{1}{1 + e^{-a(\textbf{x})}}
\end{equation}
$$

$$a(\textbf{x}) = ln\left(\frac{p(\textbf{x}|C_1)p(C_1)}{p(\textbf{x}|C_2)p(C_2)}\right)$$

This is the same equation and you can check that by substituting everything back.

And for the case of $$K\gt2$$:


$$p(C_k|\textbf{x}) = \frac{p(\textbf{x}|C_k)p(C_k)}{\sum_{j=1}^{K}p(\textbf{x}|C_j)p(C_j)}$$

We will rewrite it as:

$$
\begin{equation} \label{softmax}
    p(C_k|\textbf{x}) = \frac{e^{~a_k(\textbf{x})}}{\sum_{j=1}^{K}e^{~a_j(\textbf{x})}}
\end{equation}
$$

Where:

$$a_j(\textbf{x}) = ln\left( p(\textbf{x}|C_j)p(C_j)\right)$$

The function $$\ref{logistic}$$ is called **logistic function** and function $$\ref{softmax}$$
is called **softmax function**.

[bayes wikipedia link]: http://en.wikipedia.org/wiki/Bayes%27_theorem