# Coursera-Neural-Networks-Assignment
Programing Assignment of the Coursera Neural Networks course

Exercise 1 was done in python with numpy.

All other exercises are written in Matlab.

---------
Assignment 4 Question 11 Note:

To calculate the Z, use the idea of chung lam:
Let's give an example. Suppose we have 2 hidden units, and 3 visible units, there will be 2 ^(2+3) = 32 possible combination of hidden & visible states.
Let S(q) = (v(1,q), v(2,q),v(3,q),h(1,q),h(2,q)), where v(1,q) is the value at S(q), so it will be either 0 or 1. q is just an index, ranging from 1 to 32.
For example, we can take S(1) = (0, 0, 0, 0, 0), S(2) = (0, 0 ,0, 0, 1), ... S(32) = (1, 1, 1, 1, 1) etc, so that all binary combination of v and h are represented.
Q11 required the partition function, let's define it to be P = sum of all q of [exp(E(S(q))], where E(S(q)) = exp(sum(v(i,q)*h(j,q)*wij)), and E(S(q)) can actually be broken into four different parts:
1) For those S(q) having h1=0 and h2=0, E(S(q(h1=0, h2=0)) = 2^3
2) For those S(q) having h1=0 and h2=1, E(S(q(h1=0, h2=1)) = sum of exp(v(1,q)*w12+v(2,q)*w22+v(3,q)*w32) all q with h1=0, h2=1 = product of [1 + exp(w12)]*[1+exp(w22)]*[1+exp(w32)]
3) For those S(q) having h1=1 and h2=0, E(S(q(h1=1, h2=0)) = sum of exp(v(1,q)*w11+v(2,q)*w21+v(3,q)*w31) all q with h1=1, h2=0 = product of [1 + exp(w11)]*[1+exp(w21)]*[1+exp(w31)
4) For those S(q) having h1=1 and h2=1, E(S(q(h1=1, h2=1)) = sum of exp[v(1,q)*(w11+w12)+v(2,q)*(w21+w22)+v(3,q)*(w31+w32)] all q with h1=1, h2=1 = product of [1 + exp(w11+w12)]*[1+exp(w21+w22)]*[1+exp(w31+w32)

In sum, instead of dealing with 32 terms, we just need to consider 2^(number of hidden units) = 4 terms
