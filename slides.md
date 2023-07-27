---
title: "Introduction to SDE training and sampling"
author: "Justin T Chiu"
theme: "metropolis"
fonttheme: "default"
section-titles: false
aspectratio: 169
date: \today
---

## Goals
2. How do we sample?
\vspace{2em}
1. How do we train?

## Setup
* Sampling requires solving the reverse SDE
\vspace{2em}
* Solving the reverse SDE requires training a score model

## Sampling by solving the reverse SDE
* Reverse SDE
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x) ]dt + g(t)dw$$
* w is Brownian motion (backward in time)
* dt is a negative time increment
* We have $f$ and $g$ from the forward SDE, which is defined by user
* Need trained time-dependent score score $s(x,t) \approx \nabla_x \log p_t(x)$

## Euler-Maruyama solver
* Simplest SDE solver is Euler-Maruyama
* Discretize $[0,1]$ into $T$ steps
* Follow reverse SDE dynamics + a little Gaussian noise $z_t$ (resembles Langevin dynamics)
\begin{align*}
\delta x &\leftarrow [f(x,t) - g^2(t)s(x,t)]\delta t + g(t) \sqrt{|\delta t|}z_t\\
x &\leftarrow x + \delta x\\
t &\leftarrow t + \delta t
\end{align*}
* Langevin dynamics for comparison:
$x \leftarrow x + \text{scale} * s(x,t) + \text{other scale} * z_t$

## Euler-Maruyama + predictor corrector illustration

## Other solvers
* Other solvers don't fix a discretization
\vspace{2em}
* Have resulted in improved image generation quality

## Onto training
* The main missing piece is the score function $s(x,t) \approx \nabla_x \log p_t(x)$
\vspace{2em}
* Training looks really close to score matching

## Training: Quick review of score matching
* Start with image $x_0$
* Have $I$ noise scales $\sigma_i$ to perturb original image $x_0$
* Use score-matching to train score function at perturbed $x'$ given $x_0$ and $\sigma_i$
$$\sum_i \sigma_i^2 E_{p_{data}(x)} E_{p_{\sigma_i}}(x'|x) \|s(x', \sigma_i) - \nabla_{x'}\log p_{\sigma_i}(x'|x)\|$$
\vspace{4em}

## SDE score training
* Score matching objective
$$\sum_i \sigma_i^2 E_{p_{data}(x)} E_{p_{\sigma_i}}(x'|x) \|s(x', \sigma_i) - \nabla_{x'}\log p_{\sigma_i}(x'|x)\|$$
* SDE training objective
$$E_{t\sim U(0,1)}E_{x(0)}E_{x(t)|x(0)} \lambda(t) \cdot \|s(x,t) - \nabla_x \log p_{0t}(x(t)|x(0))\|_2^2$$
* How do we get $x(t) | x(0)$? Solve forward noising SDE, which was manually defined w/o learnable components
