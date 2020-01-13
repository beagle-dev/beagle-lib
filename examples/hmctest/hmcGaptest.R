# This file helps to calculate the same likelihood as in hmctest.cpp
# Xiang Ji
# xji3@ncsu.edu

rm(list=ls())  # clean up workspace
getLoglikelihood <- function(data, rate.param, blen.param, stationary.dist, rates = NULL, weights = NULL){
  library(Matrix)

  data.1 <- data$data.1
  data.2 <- data$data.2
  data.3 <- data$data.3

  pi.A <- rate.param$pi.A
  pi.C <- rate.param$pi.C
  pi.G <- rate.param$pi.G
  pi.T <- rate.param$pi.T
  kappa <- rate.param$kappa

  Q <- matrix(c(0.0, pi.C, kappa * pi.G, pi.T,
                pi.A, 0.0, pi.G, kappa*pi.T,
                kappa*pi.A, pi.C, 0.0, pi.T,
                pi.A, kappa*pi.C, pi.G, 0.0), 4, 4, byrow=TRUE)
  Q <- Q - diag(rowSums(Q))
  # Normalize Q matrix to have unit being expected number of changes per site
  Q.normalized <- Q / sum(-stationary.dist * diag(Q))
  Q <- Q.normalized

  if(is.null(rates) | is.null(weights)){
    rates = c(1.0)
    weights = c(1.0)
  }

  likelihood.mat <- NULL
  for(i in 1:length(rates)){
    ri = rates[i]
    wi = weights[i]

    blen.A3 <- blen.param$blen.A3 * ri
    blen.A2 <- blen.param$blen.A2 * ri
    blen.BA <- blen.param$blen.BA * ri
    blen.B1 <- blen.param$blen.B1 * ri

    Ptr.A3 <- expm(Q*blen.A3)
    Ptr.A2 <- expm(Q*blen.A2)
    Ptr.BA <- expm(Q*blen.BA)
    Ptr.B1 <- expm(Q*blen.B1)

    # Now calculate left (#.1) and right (#.2) post-order (conditional likelihood) of each internal node
    A.1 <- Ptr.A3 %*% data.3
    A.2 <- Ptr.A2 %*% data.2
    A.post.order <- A.1 * A.2
    B.1 <- Ptr.BA %*% A.post.order
    B.2 <- Ptr.B1 %*% data.1
    B.post.order <- B.1 * B.2

    likelihood.mat <- rbind(likelihood.mat, wi*colSums(stationary.dist * B.post.order))
  }

  return((log(colSums(likelihood.mat))))
}

# Two rate categories
rates = c(0.14251623900062188, 1.857483760999378)
weights = c(0.5, 0.5)

# One rate category
# rates = c(1.0)
# weights = c(1.0)

library(Matrix)
# Define branch lengths first
blen.A3 <- 0.6
blen.A2 <- 0.6
blen.BA <- 0.7
blen.B1 <- 1.3

# Define data vectors at tips
# data.1 = "AAATC"
# data.2 = "GAGTC"
# data.3 = "GAGGC"
data.1 <- matrix(c(1., 0., 0., 0.,
                   1., 0., 0., 0.,
                   1., 1., 1., 1.,
                   0., 0., 0., 1.,
                   0., 1., 0., 0.), 4, 5)
data.2 <- matrix(c(1., 1., 1., 1.,
                   1., 0., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.,
                   0., 1., 0., 0.), 4, 5)
data.3 <- matrix(c(0., 0., 1., 0.,
                   1., 0., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 1., 0.,
                   0., 1., 0., 0.), 4, 5)

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

# Now define stationary nucleotide frequency
# <parameter id="hky.frequencies" value="0.1 0.3 0.2 0.4"/>
#
pi.A <- 0.1
pi.C <- 0.3
pi.G <- 0.2
pi.T <- 0.4
kappa <- 1.0

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

A.gradient.mat <- NULL
tip.1.gradient.mat <- NULL
tip.2.gradient.mat <- NULL
tip.3.gradient.mat <- NULL
likelihood.mat <- NULL

A.pre.order.list <- NULL
B.pre.order.list <- NULL
tip.1.pre.order.list <- NULL
tip.2.pre.order.list <- NULL
tip.3.pre.order.list <- NULL
for(i in 1:length(rates)){
  # Ptr matrices
  Ptr.A3 <- evec %*% diag(exp(eval * blen.A3 * rates[i])) %*% ivec
  Ptr.A2 <- evec %*% diag(exp(eval * blen.A2 * rates[i])) %*% ivec
  Ptr.BA <- evec %*% diag(exp(eval * blen.BA * rates[i])) %*% ivec
  Ptr.B1 <- evec %*% diag(exp(eval * blen.B1 * rates[i])) %*% ivec

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

  likelihood.mat <- rbind(likelihood.mat, (colSums(stationary.dist * B.post.order)))

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

  B.pre.order.list[[i]] <- t(B.pre.order)
  A.pre.order.list [[i]] <- t(A.pre.order)
  tip.1.pre.order.list[[i]] <- t(tip.1.pre.order)
  tip.3.pre.order.list[[i]] <- t(tip.3.pre.order)
  tip.2.pre.order.list[[i]] <- t(tip.2.pre.order)

  # Now caculate branch length gradient for 1st site
  A.gradient <- NULL
  tip.1.gradient <- NULL
  tip.2.gradient <- NULL
  tip.3.gradient <- NULL
  for(i in 1:dim(data.1)[2]){
    A.gradient <- c(A.gradient, A.post.order[, i] %*% t(Q.normalized) %*% A.pre.order[, i] / sum(A.post.order[, i] * A.pre.order[, i]))
    tip.1.gradient <- c(tip.1.gradient, data.1[, i] %*% t(Q.normalized) %*% tip.1.pre.order[, i] / sum(data.1[, i] * tip.1.pre.order[, i]))
    tip.2.gradient <- c(tip.2.gradient, data.2[, i] %*% t(Q.normalized) %*% tip.2.pre.order[, i] / sum(data.2[, i] * tip.2.pre.order[, i]))
    tip.3.gradient <- c(tip.3.gradient, data.3[, i] %*% t(Q.normalized) %*% tip.3.pre.order[, i] / sum(data.3[, i] * tip.3.pre.order[, i]))
  }

  A.gradient.mat <- rbind(A.gradient.mat, A.gradient)
  tip.1.gradient.mat <- rbind(tip.1.gradient.mat, tip.1.gradient)
  tip.2.gradient.mat <- rbind(tip.2.gradient.mat, tip.2.gradient)
  tip.3.gradient.mat <- rbind(tip.3.gradient.mat, tip.3.gradient)
}


data <- list(data.1 = data.1, data.2 = data.2, data.3 = data.3)
rate.param <- list(pi.A = pi.A, pi.C = pi.C, pi.G = pi.G, pi.T = pi.T, kappa = kappa)
blen.param <- list(blen.A2 = blen.A2, blen.A3 = blen.A3, blen.B1 = blen.B1, blen.BA = blen.BA)
ll <- getLoglikelihood(data, rate.param, blen.param, stationary.dist, rates, weights)

print(B.pre.order.list)
# now numerically calculate blen derivatives
dl <- 1e-7
blen.param.dl <- list(blen.A2 = blen.A2, blen.A3 = blen.A3, blen.B1 = blen.B1, blen.BA = blen.BA * (1+dl))
ll.dl <- getLoglikelihood(data, rate.param, blen.param.dl, stationary.dist, rates, weights)
A.gradient.numerical <- (ll.dl - ll)/(blen.BA * dl)
A.gradient <- colSums(A.gradient.mat * (weights * rates * likelihood.mat))/colSums(weights * likelihood.mat)
print(A.pre.order.list)

blen.param.dl <- list(blen.A2 = blen.A2, blen.A3 = blen.A3, blen.B1 = blen.B1 * (1+dl), blen.BA = blen.BA)
tip.1.gradient.numerical <- (getLoglikelihood(data, rate.param, blen.param.dl, stationary.dist, rates, weights) - ll)/(blen.B1 * dl)
tip.1.gradient <- colSums(tip.1.gradient.mat * (weights * rates * likelihood.mat))/colSums(weights * likelihood.mat)
print(tip.1.pre.order.list)

blen.param.dl <- list(blen.A2 = blen.A2, blen.A3 = blen.A3 * (1+dl), blen.B1 = blen.B1, blen.BA = blen.BA)
tip.3.gradient.numerical <- (getLoglikelihood(data, rate.param, blen.param.dl, stationary.dist, rates, weights) - ll)/(blen.A3 * dl)
tip.3.gradient <- colSums(tip.3.gradient.mat * (weights * rates * likelihood.mat))/colSums(weights * likelihood.mat)
print(tip.3.pre.order.list)

blen.param.dl <- list(blen.A2 = blen.A2 * (1+dl), blen.A3 = blen.A3, blen.B1 = blen.B1, blen.BA = blen.BA)
tip.2.gradient.numerical <- (getLoglikelihood(data, rate.param, blen.param.dl, stationary.dist, rates, weights) - ll)/(blen.A2 * dl)
tip.2.gradient <- colSums(tip.2.gradient.mat * (weights * rates * likelihood.mat))/colSums(weights * likelihood.mat)
print(tip.2.pre.order.list)
cat("logL = ", formatC(signif(sum(log(colSums(likelihood.mat * weights))),digits=18), digits=16,format="fg", flag="#"))


cat("Gradient for branch (of node) 1: \n   ", tip.3.gradient.numerical, " (numerical)\n   ", tip.3.gradient, " \n")
cat("Gradient for branch (of node) 0: \n   ", tip.2.gradient.numerical, " (numerical)\n   ", tip.2.gradient, " \n")
cat("Gradient for branch (of node) 2: \n   ", tip.1.gradient.numerical, " (numerical)\n   ", tip.1.gradient, " \n")
cat("Gradient for branch (of node) 3: \n   ", A.gradient.numerical, " (numerical)\n   ", A.gradient, " \n")
