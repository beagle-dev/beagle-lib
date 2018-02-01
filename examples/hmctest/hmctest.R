# This file helps to calculate the same likelihood as in hmctest.cpp
# Xiang Ji
# xji3@ncsu.edu

rm(list=ls())  # clean up workspace
library(Matrix)
# Define branch lengths first
blen.A3 <- 0.6
blen.A2 <- 0.6
blen.BA <- 0.7
blen.B1 <- 1.3

# Define data vectors at tips
# data.1 = "AAAT"
# data.2 = "GAGT"
# data.3 = "GAGG"
data.1 <- matrix(c(1., 0., 0., 0., 
                   1., 0., 0., 0.,
                   1., 0., 0., 0.,
                   0., 0., 0., 1.), 4, 4)
data.2 <- matrix(c(0., 0., 1., 0.,
                   1., 0., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.), 4, 4)
data.3 <- matrix(c(0., 0., 1., 0., 
                   1., 0., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 1., 0.), 4, 4)

# add-in eigen decomposition
evec <- matrix(c(0.9819805,  0.040022305,  0.04454354,  -0.5,
                 -0.1091089, -0.002488732, 0.81606029,  -0.5,
                 -0.1091089, -0.896939683, -0.11849713, -0.5,
                 -0.1091089,  0.440330814, -0.56393254, -0.5), 4, 4, byrow=T)

ivec <- matrix(c(0.9165151, -0.3533241, -0.1573578, -0.4058332,
                 0.0,  0.2702596, -0.8372848,  0.5670252,
                 0.0,  0.8113638, -0.2686725, -0.5426913,
                 -0.2, -0.6, -0.4, -0.8), 4, 4, byrow = T)

eval <- c( -1.428571, -1.428571, -1.428571, 0.0)

#JC69
# evec <- matrix(c(1.0,  2.0,  0.0,  0.5,
#         1.0,  -2.0,  0.5,  0.0,
#         1.0,  2.0, 0.0,  -0.5,
#         1.0,  -2.0,  -0.5,  0.0), 4, 4, byrow=T)
# 
# ivec <- matrix(c(0.25,  0.25,  0.25,  0.25,
#         0.125,  -0.125,  0.125,  -0.125,
#         0.0,  1.0,  0.0,  -1.0,
#         1.0,  0.0,  -1.0,  0.0), 4, 4, byrow = T)
# 
# eval <- c( 0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333)

# Now define stationary nucleotide frequency
# <parameter id="hky.frequencies" value="0.1 0.3 0.2 0.4"/>
# 
pi.A <- 0.1
pi.C <- 0.3
pi.G <- 0.2
pi.T <- 0.4
kappa <- 1.0

# pi.A <- 0.25
# pi.C <- 0.25
# pi.G <- 0.25
# pi.T <- 0.25

# stationary dist
stationary.dist <- c(pi.A, pi.C, pi.G, pi.T)
# Now construct rate matrix Q
Q <- matrix(c(0.0, pi.C, kappa * pi.G, pi.T, 
              pi.A, 0.0, pi.G, kappa*pi.T,
              kappa*pi.A, pi.C, 0.0, pi.T,
              pi.A, kappa*pi.C, pi.G, 0.0), 4, 4, byrow=TRUE)
Q <- Q - diag(rowSums(Q))

# Normalize Q matrix to have unit being expected number of changes per site
Q.normalized <- Q / sum(-stationary.dist * diag(Q))
Q.normalized <- evec %*% diag(eval) %*% ivec

# update Q by the normalized matrix, this step is just for sanity check that can be commented out
Q <- Q.normalized

# Ptr matrices
Ptr.A3 <- evec %*% diag(exp(eval * blen.A3)) %*% ivec
Ptr.A2 <- evec %*% diag(exp(eval * blen.A2)) %*% ivec
Ptr.BA <- evec %*% diag(exp(eval * blen.BA)) %*% ivec
Ptr.B1 <- evec %*% diag(exp(eval * blen.B1)) %*% ivec

# Now calculate left (#.1) and right (#.2) post-order (conditional likelihood) of each internal node
A.1 <- Ptr.A3 %*% data.3
A.2 <- Ptr.A2 %*% data.2
A.post.order <- A.1 * A.2
B.1 <- Ptr.BA %*% A.post.order
B.2 <- Ptr.B1 %*% data.1
B.post.order <- B.1 * B.2

# Now show the posterior probability of node B
B.posterior <- B.post.order * stationary.dist
B.posterior/sum(B.posterior)

# log-likelihood
print(log(colSums(stationary.dist * B.post.order)))
print(sum(log(colSums(stationary.dist * B.post.order))))

# Now calculate the pre-order traversals
B.pre.order <- stationary.dist %*% matrix(1., 1, dim(data.1)[2])
A.pre.order <- crossprod(Ptr.BA, (B.pre.order * B.2))
#A.pre.order <- t(t(A.pre.order) / colSums(A.pre.order))

# Now update the pre-order partials on tips
tip.1.pre.order <- crossprod(Ptr.B1, B.pre.order*B.1)
#tip.1.pre.order <- t(t(tip.1.pre.order) / colSums(tip.1.pre.order))

tip.2.pre.order <- crossprod(Ptr.A2, A.pre.order*A.1)
#tip.2.pre.order <- t(t(tip.2.pre.order) / colSums(tip.2.pre.order))

tip.3.pre.order <- crossprod(Ptr.A3, A.pre.order*A.2)
#tip.3.pre.order <- t(t(tip.3.pre.order) / colSums(tip.3.pre.order))

# output pre-order partials to screen in the same order as hmctest.cpp
t(B.pre.order)
t(A.pre.order)
t(tip.1.pre.order)
t(tip.3.pre.order)
t(tip.2.pre.order)

# output log-likelihood again
print(sum(log(colSums(stationary.dist * B.post.order))))
