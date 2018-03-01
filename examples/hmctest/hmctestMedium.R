# This file helps to calculate the same likelihood as in hmctest.cpp
# Xiang Ji
# xji3@ncsu.edu

rm(list=ls())  # clean up workspace
getLoglikelihood <- function(data, rate.param, blen.param, stationary.dist, rates = NULL, weights = NULL){
  library(Matrix)
  
  data.0 <- data$data.0
  data.1 <- data$data.1
  data.2 <- data$data.2
  data.3 <- data$data.3
  data.4 <- data$data.4
  
  
  pi.A <- rate.param$pi.A
  pi.C <- rate.param$pi.C
  pi.G <- rate.param$pi.G
  pi.T <- rate.param$pi.T
  kappa <- rate.param$kappa
  # stationary dist
  stationary.dist <- c(pi.A, pi.C, pi.G, pi.T)
  
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
    
    blen.50 <- blen.param$blen.50 * ri
    blen.51 <- blen.param$blen.51 * ri
    blen.62 <- blen.param$blen.62 * ri
    blen.63 <- blen.param$blen.63 * ri
    blen.75 <- blen.param$blen.75 * ri
    blen.76 <- blen.param$blen.76 * ri
    blen.87 <- blen.param$blen.87 * ri
    blen.84 <- blen.param$blen.84 * ri

    Ptr.50 <- expm(Q*blen.50)
    Ptr.51 <- expm(Q*blen.51)
    Ptr.62 <- expm(Q*blen.62)
    Ptr.63 <- expm(Q*blen.63)
    Ptr.75 <- expm(Q*blen.75)
    Ptr.76 <- expm(Q*blen.76)
    Ptr.87 <- expm(Q*blen.87)
    Ptr.84 <- expm(Q*blen.84)
    
    # Now calculate left (#.1) and right (#.2) post-order (conditional likelihood) of each internal node
    post.5.left <- Ptr.50 %*% data.0
    post.5.right <- Ptr.51 %*% data.1
    post.5 <- post.5.left * post.5.right
    post.6.left <- Ptr.62 %*% data.2
    post.6.right <- Ptr.63 %*% data.3
    post.6 <- post.6.left * post.6.right
    post.7.left <- Ptr.75 %*% post.5
    post.7.right <- Ptr.76 %*% post.6
    post.7 <- post.7.left * post.7.right
    post.8.left <- Ptr.87 %*% post.7
    post.8.right <- Ptr.84 %*% data.4
    post.8 <- post.8.left * post.8.right
    
    likelihood.mat <- rbind(likelihood.mat, wi*colSums(stationary.dist * post.8))
  }
  
  return(sum(log(colSums(likelihood.mat))))
}

getData <- function(seq.input){
  seq.separate <- strsplit(seq.input, "")[[1]]
  mat.data <- NULL
  for(i in 1:length(seq.separate)){
    if(seq.separate[i] == "A"){
      mat.data <- c(mat.data, 1, 0, 0, 0)
    }else if(seq.separate[i] == "C"){
      mat.data <- c(mat.data, 0, 1, 0, 0)
    }else if(seq.separate[i] == "G"){
      mat.data <- c(mat.data, 0, 0, 1, 0)
    }else if(seq.separate[i] == "T"){
      mat.data <- c(mat.data, 0, 0, 0, 1)
    }
  }
  return(matrix(mat.data, 4, length(seq.separate)))
}

library(Matrix)

D4Brazi82 <- "ATGCGATGCGTAGGAGTAGGAAACAGAGACTTTGTGGAAGGAGTCTCAGGTGGAGCATGGGTCGACCTGGTGCTAGAACATGGAGGATGCGTCACAACCATGGCCCAGGGAAAACCAACCTTGGATTTTGAACTGACCAAGACAACAGCCAAGGAAGTGGCTCTGTTAAGAACCTATTGCATTGAAGCCTCAATATCAAACATAACTACGGCAACAAGATGTCCAACGCAAGGAGAGCCTTATCTGAAAGAGGAACAGGACCAACAGTACATTTGCCGGAGAGATGTGGTAGACAGAGGGTGGGGCAATGGCTGTGGCTTGTTTGGAAAAGGAGGAGTTGTGACATGTGCGAAGTTTTCATGTTCGGGGAAGATAACAGGCAATTTGGTCCAAATTGAGAACCTTGAATACACAGTGGTTGTAACAGTCCACAATGGAGACACCCATGCAGTAGGAAATGACACATCCAATCATGGAGTTACAGCCATGATAACTCCCAGGTCACCATCGGTGGAAGTCAAATTGCCGGACTATGGAGAACTAACACTCGATTGTGAACCCAGGTCTGGAATTGACTTTAATGAGATGATTCTGATGAAAATGAAAAAGAAAACATGGCTCGTGCATAAGCAATGGTTTTTGGATCTGCCTCTTCCATGGACAGCAGGAGCAGACACATCAGAGGTTCACTGGAATTACAAAGAGAGAATGGTGACATTTAAGGTTCCTCATGCCAAGAGACAGGATGTGACAGTGCTGGGATCTCAGGAAGGAGCCATGCATTCTGCCCTCGCTGGAGCCACAGAAGTGGACTCCGGTGATGGAAATCACATGTTTGCAGGACATCTCAAGTGCAAAGTCCGTATGGAGAAATTGAGAATCAAGGGAATGTCATACACGATGTGTTCAGGAAAGTTTTCAATTGACAAAGAGATGGCAGAAACACAGCATGGGACAACAGTGGTGAAAGTCAAGTATGAAGGTGCTGGAGCTCCGTGTAAAGTCCCCATAGAGATAAGAGATGTAAACAAGGAAAAAGTGGTTGGGCGTATCATCTCATCCACCCCTTTGGCTGAGAATACCAACAGTGTAACCAACATAGAATTAGAACCCCCCTTTGGGGACAGCTACATAGTGATAGGTGTTGGAAACAGCGCATTAACACTCCATTGGTTCAGGAAAGGGAGTTCCATTGGCAAGATGTTTGAGTCCACATACAGAGGTGCAAAACGAATGGCCATTCTAGGTGAAACAGCTTGGGATTTTGGTTCCGTTGGTGGATTGTTCACATCATTGGGAAAGGCTGTGCACCAGGTTTTTGGAAGTGTGTATACAACCATGTTTGGAGGAGTCTCATGGATGATTAGAATCCTAATTGGGTTCTTAGTGTTGTGGATTGGCACGAACTCAAGGAACACTTCAATGGCTATGACGTGCATAGCTGTTGGAGGAATCACTCTGTTTCTGGGCTTCACAGTTCAAGCA"
D4ElSal83 <- "ATGCGATGCGTAGGAGTAGGAAACAGAGACTTTGTGGAAGGAGTCTCAGGTGGAGCATGGGTCGACCTGGTGCTAGAACATGGAGGATGCGTCACAACCATGGCCCAGGGAAAACCAACCTTGGATTTTGAACTGACTAAGACAACAGCCAAGGAAGTGGCTCTGTTAAGAACCTATTGCATTGAAGCCTCAATATCAAACATAACTACGGCAACAAGATGTCCAACGCAAGGAGAGCCTTATCTGAAAGAGGAACAGGACCAACAGTACATTTGCCGGAGAGATGTGGTAGACAGAGGGTGGGGCAATGGCTGTGGCTTGTTTGGAAAAGGAGGAGTTGTGACATGTGCGAAGTTTTCATGTTCGGGGAAGATAACAGGCAATTTGGTCCAAATTGAGAACCTTGAATACACAGTGGTTGTAACAGTCCACAATGGAGACACCCATGCAGTAGGAAATGACACATCCAATCATGGAGTTACAGCCATGATAACTCCCAGGTCACCATCGGTGGAAGTCAAATTGCCGGACTATGGAGAACTAACACTCGATTGTGAACCCAGGTCTGGAATTGACTTTAATGAGATGATTCTGATGAAAATGAAAAAGAAAACATGGCTCGTGCATAAGCAATGGTTTTTGGATCTGCCTCTTCCATGGACAGCAGGAGCAGACACATCAGAGGTTCACTGGAATTACAAAGAGAGAATGGTGACATTTAAGGTTCCTCATGCCAAGAGACAGGATGTGACAGTGCTGGGATCTCAGGAAGGAGCCATGCATTCTGCCCTCGCTGGAGCCACAGAAGTGGACTCCGGTGATGGAAATCATATGTTTGCAGGACATCTCAAGTGCAAAGTCCGTATGGAGAAATTGAGAATCAAGGGAATGTCATACACGATGTGTTCAGGAAAGTTTTCAATTGACAAAGAGATGGCAGAAACACAGCATGGGACAACAGTGGTGAAAGTCAAGTATGAAGGTGCTGGAGCTCCGTGTAAAGTCCCCATAGAGATAAGAGATGTAAACAAGGAAAAAGTGGTTGGGCGTATCATCTCATCCACCCCTTTGGCTGAGAATACCAACAGTGTAACCAACATAGAATTAGAACCCCCCTTTGGGGACAGCTACATAGTGATAGGTGTTGGAAACAGCGCATTAACACTCCATTGGTTCAGGAAAGGGAGTTCCATTGGCAAGATGTTTGAGTCCACATACAGAGGTGCAAAACGAATGGCCATTCTAGGTGAAACAGCTTGGGATTTTGGTTCCGTTGGTGGACTGTTCACATCATTGGGAAAGGCTGTGCACCAGGTTTTTGGAAGTGTGTATACAACCATGTTTGGAGGAGTCTCATGGATGATTAGAATCCTAATTGGGTTCTTAGTGTTGTGGATTGGCACGAACTCAAGGAACACTTCAATGGCTATGACGTGCATAGCTGTTGGAGGAATCACTCTGTTTCTGGGCTTCACAGTTCAAGCA"
D4ElSal94 <- "ATGCGATGCGTAGGAGTAGGAAACAGAGACTTTGTGGAAGGAGTCTCAGGTGGAGCATGGGTCGACCTGGTGCTAGAACATGGAGGATGCGTCACAACCATAGCCCAGGGAAAACCAACCTTGGATTTTGAATTGACTAAGACAACAGCCAAGGAAGTGGCTCTGTTAAGAACCTATTGCATTGAAGCCTCAATATCAAACATAACTACGGCAACAAGATGTCCAACGCAAGGAGAGCCTTATCTGAAAGAGGAACAGGACCAACAGTACATTTGCCGGAGAGATGTGGTAGACAGAGGGTGGGGGAATGGCTGTGGCTTGCTTGGAAAAGGAGGAGTTGTGACATGTGCGAAGTTTTCATGTTCGGGGAAGATAACAGGCAATTTGGTCCAAATTGAGAACCTTGAATACACAGTGGTTGTAACAGTCCACAATGGAGATACCCATGCAGTAGGAAATGACACATCCAATCATGGAGTTACAGCCACGATAACTCCCAGGTCACCATCGGTGGAAGTCAAATTGCCGGACTATGGAGAACTAACACTCGATTGTGAACCCAGATCTGGAATTGATTTTAATGAGATGATTCTGATGAAAATGAAAAAGAAAACATGGCTCGTGCATAAGCAATGGTTTTTGGATCTGCCTCTTCCATGGACAGCAGGAGCAGACACATCAGAGGTTCACTGGAATTACAAAGAGAGAATGGTGACATTCAAGGTTCCTCATGCCAAGAGACAGGATGTGACAGTGCTGGGATCTCAGGAAGGAGCCATGCATTCTGCCCTCGCTGGAGCCACAGAAGTGGACTCCGGTGATGGAAATCACATGTTTGCAGGACATCTCAAGTGCAAAGTCCGCATGGAGAAATTGAGAATCAAGGGAATGTCATACACGATGTGTTCAGGAAAGTTTTCAATTGATAAAGAGATGGCAGAAACACAGCATGGGACAACAGTGGTGAAAGTCAAGTATGAAGGTGCTGGAGCTCCGTGTAAAGTCCCCATAGAGATAAGAGATGTAAACAAGGAAAAAGTGGTTGGGCGTATCATCTCATCCACCCCTTTGGCTGAGAATACCAACAGTGTAACCAACATAGAATTAGAACCCCCCTTTGGGGACAGCTACATAGTGATAGGTGTCGGAAACAGCGCATTAACACTCCATTGGTTCAGGAAAGGGAGTTCCATTGGCAAGATGTTTGAGTCCACATACAGAGGTGCAAAACGAATGGCCATTCTAGGTGAAACAGCTTGGGATTTTGGTTCCGTTGGTGGACTGTTCACATCATTGGGAAAGGCTGTGCACCAGGTTTTTGGAAGTGTGTACACAACCATGTTTGGAGGAGTCTCATGGATGATTAGAATCCTAATTGGGTTCTTAGTGTTATGGATTGGCACGAACTCAAGGAACACTTCAATGGCTATGACGTGCATAGCTGTTGGAGGAATCACTCTGTTTCTGGGCTTCACAGCTCAAGCA"
D4Indon76 <- "ATGCGATGCGTAGGAGTAGGAAACAGAGACTTTGTGGAAGGAGTCTCAGGTGGAGCATGGGTCGATCTGGTGCTAGAACATGGAGGATGCGTCACAACCATGGCCCAGGGAAAACCAACCTTGGATTTTGAACTGACTAAGACAACAGCCAAGGAAGTGGCTCTGTTAAGAACCTATTGCATTGAAGCCTCAATATCAAACATAACCACGGCAACAAGATGTCCAACGCAAGGAGAGCCTTATCTAAAAGAGGAACAAGACCAACAGTACATTTGCCGGAGAGATGTGGTAGACAGAGGGTGGGGCAATGGCTGTGGCTTGTTTGGAAAAGGAGGAGTTGTGACATGTGCGAAGTTTTCATGTTCGGGGAAGATAACAGGCAATTTGGTCCAAATTGAGAACCTTGAATACACAGTGGTTGTAACAGTCCACAATGGAGACACCCATGCAGTAGGAAATGACACATCCAATCATGGAGTTACAGCCACGATAACTCCCAGGTCACCATCGGTGGAAGTCAAATTGCCGGACTATGGAGAACTAACACTCGATTGTGAACCCAGGTCTGGAATTGACTTTAATGAGATGATTCTGATGAAAATGAAAAAGAAAACATGGCTTGTGCATAAGCAATGGTTTTTGGATCTACCTCTACCATGGACAGCAGGAGCAGACACATCAGAGGTTCACTGGAATTACAAAGAGAGAATGGTGACATTTAAGGTTCCTCATGCCAAGAGACAGGATGTGACAGTGCTGGGATCTCAGGAAGGAGCCATGCATTCTGCCCTCGCTGGAGCCACAGAAGTGGACTCCGGTGATGGAAATCACATGTTTGCAGGACATCTCAAGTGCAAAGTCCGTATGGAGAAATTGAGAATCAAGGGAATGTCATACACGATGTGTCCAGGAAAGTTCTCAATTGACAAAGAGATGGCAGAAACACAGCATGGGACAACAGTGGTGAAAGTCAAGTATGAAGGTGCTGGAGCTCCGTGTAAAGTCCCCATAGAGATAAGAGATGTGAACAAGGAAAAAGTGGTTGGGCGTATCATCTCATCCACCCCTTTGGCTGAGAATACCAACAGTGCAACCAACATAGAGTTAGAACCCCCCTTTGGGGACAGCTACATAGTGATAGGTGTTGGAAACAGTGCATTAACACTCCATTGGTTCAGGAAAGGGAGTTCCATTGGCAAGATGTTTGAGTCCACATACAGAGGTGCAAAACGAATGGCCATTCTAGGTGAAACAGCTTGGGATTTTGGTTCCGTTGGTGGACTGCTCACATCATTGGGAAAGGCTGTGCACCAGGTTTTTGGAAGTGTGTATACAACCATGTTTGGAGGAGTCTCATGGATGATTAGAATCCTAATTGGGTTCCTAGTGTTGTGGATTGGCACGAACTCAAGGAACACTTCAATGGCTATGACGTGCATAGCTGTTGGAGGAATCACTCTGTTTCTGGGCTTCACAGTTCAAGCA"
D4Indon77 <- "ATGCGATGCGTAGGAGTAGGAAACAGAGACTTTGTGGAAGGAGTCTCAGGTGGAGCATGGGTCGATCTGGTGCTAGAACATGGAGGATGCGTCACAACCATGGCCCAGGGAAAACCAACCTTGGATTTTGAACTGACTAAGACAACAGCCAAGGAAGTGGCTCTGTTAAGAACCTATTGCATTGAAGCCTCAATATCAAACATAACCACGGCAACAAGATGTCCAACGCAAGGAGAGCCTTATCTAAAAGAGGAACAAGACCAACAGTACATTTGCCGGAGAGATGTGGTAGACAGAGGGTGGGGCAATGGCTGTGGCTTGTTTGGAAAAGGAGGAGTTGTGACATGTGCGAAGTTTTCATGTTCGGGGAAGATAACAGGCAATTTGGTCCAAATTGAGAACCTTGAATACACAGTAGTTGTAACAGTCCACAATGGAGACACCCATGCAGTAGGAAATGACACATCCAACCATGGAGTTACAGCCACGATAACTCCCAGGTCACCATCGGTGGAAGTCAAATTGCCGGACTATGGAGAACTAACACTCGATTGTGAACCCAGGTCTGGAATTGACTTTAATGAGATGATTCTGATGAAAATGAAAAAGAAAACATGGCTTGTGCATAAGCAATGGTTTTTGGATCTACCTCTACCATGGACAGCAGGAGCAGACACATCAGAGGTTCACTGGAATTACAAAGAGAGAATGGTGACATTTAAGGTTCCTCATGCCAAGAGACAGGATGTGACAGTGCTGGGATCTCAGGAAGGAGCCATGCATTCTGCCCTCGCTGGAGCCACAGAAGTGGACTCCGGTGATGGAAATCACATGTTTGCAGGACATCTCAAGTGCAAAGTCCGTATGGAGAAATTGAGAATCAAGGGAATGTCATACACGATGTGTTCAGGAAAGTTCTCAATTGACAAAGAGATGGCAGAAACACAGCATGGGACAACAGTGGTGAAAGTCAAGTATGAAGGTGCTGGAGCTCCGTGCAAAGTCCCCATAGAGATAAGAGATGTAAACAAGGAAAAAGTGGTTGGGCGTATCATCTCATCCACCCCTTTGGCTGAGAATACCAACAGTGTAACCAACATAGAATTAGAACCCCCCTTTGGGGACAGCTACATAGTGATAGGTGTTGGAAACAGTGCATTAACACTCCATTGGTTCAGGAAAGGGAGTTCCATTGGCAAGATGTTTGAGTCCACATACAGAGGTGCAAAACGAATGGCCATTCTAGGTGAAACAGCTTGGGATTTTGGTTCCGTTGGTGGACTGTTCACATCATTGGGAAAGGCTGTGCACCAGGTTTTTGGAAGTGTGTATACAACCATGTTTGGAGGAGTCTCATGGATGATTAGAATCCTAATTGGCTTCTTAGTGTTGTGGATTGGCACGAACTCAAGGAACACTTCAATGGCTATGACGTGCATAGCTGTTGGAGGAATCACTCTGTTTCTGGGCTTCACAGTTCAAGCA"

data.0 <- getData(D4ElSal94)
data.1 <- getData(D4Indon76)
data.2 <- getData(D4Brazi82)
data.3 <- getData(D4ElSal83)
data.4 <- getData(D4Indon77)

post.0 <- data.0
post.1 <- data.1
post.2 <- data.2
post.3 <- data.3
post.4 <- data.4

# blen.50 <- 25.81403421468474
# blen.51 <- 7.814034214684739
# blen.62 <- 36.80326223293307
# blen.63 <- 37.80326223293307
# blen.84 <- 282.8618556834007
# blen.75 <- 244.56614874230695
# blen.76 <- 221.57692072405862
# blen.87 <- 29.481672726408988

blen.50 <- 0.81403421468474
blen.51 <- 0.814034214684739
blen.62 <- 0.80326223293307
blen.63 <- 0.80326223293307
blen.84 <- 0.8618556834007
blen.75 <- 0.56614874230695
blen.76 <- 0.57692072405862
blen.87 <- 0.481672726408988
# Now define stationary nucleotide frequency
# <parameter id="hky.frequencies" value="0.1 0.3 0.2 0.4"/>
# 
pi.A <- 0.1
pi.C <- 0.3
pi.G <- 0.2
pi.T <- 0.4
kappa <- 1.0

data <- list(data.0 = data.0, data.1 = data.1, data.2 = data.2, data.3 = data.3, data.4 = data.4)
rate.param <- list(pi.A = pi.A, pi.C = pi.C, pi.G = pi.G, pi.T = pi.T, kappa = kappa)
blen.param <- list(blen.50 = blen.50, blen.51 = blen.51, blen.62 = blen.62, blen.63 = blen.63,
                   blen.84 = blen.84, blen.75 = blen.75, blen.76 = blen.76, blen.87 = blen.87)

# Two rate categories
# rates = c(3. * 1:2 / 5.)
# weights = c(1:2 / 3.)
rates = c(0.14251623900062188, 1.85748376099937812)
weights = c(0.5, 0.5)


# One rate category
# rates = c(1.0)
# weights = c(1.0)

# # # Four rate categories
# # library(truncdist)
# # rates <- NULL
# # for(rate.iter in 1:4){
# #   lower.bound <- qgamma((rate.iter - 1)/4, 0.5, 0.5)
# #   upper.bound <- qgamma(rate.iter/4, 0.5, 0.5)
# #   rates <- c(rates, extrunc(spec="gamma", lower.bound, upper.bound, 0.5, 0.5))
# # }
# rates = c(0.02907775442778477,
#           0.28071453392572127,
#           0.9247730548197041,
#           2.76543465682679)
# weights = c(0.25, 0.25, 0.25, 0.25)

# # # Five rate categories
# library(truncdist)
# rates <- NULL
# for(rate.iter in 1:5){
#   lower.bound <- qgamma((rate.iter - 1)/5, 0.5, 0.5)
#   upper.bound <- qgamma(rate.iter/5, 0.5, 0.5)
#   rates <- c(rates, extrunc(spec="gamma", lower.bound, upper.bound, 0.5, 0.5))
# }
# weights = c(0.2, 0.2, 0.2, 0.2, 0.2)

# Define data vectors at tips

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

likelihood.mat <- NULL
node.0.pre.order.list <- NULL
node.1.pre.order.list <- NULL
node.2.pre.order.list <- NULL
node.3.pre.order.list <- NULL
node.4.pre.order.list <- NULL
node.5.pre.order.list <- NULL
node.6.pre.order.list <- NULL
node.7.pre.order.list <- NULL

node.0.gradient.mat <- NULL
node.1.gradient.mat <- NULL
node.2.gradient.mat <- NULL
node.3.gradient.mat <- NULL
node.4.gradient.mat <- NULL
node.5.gradient.mat <- NULL
node.6.gradient.mat <- NULL
node.7.gradient.mat <- NULL
node.8.pre.order <- stationary.dist %*% matrix(1., 1, dim(data.0)[2])
for(i in 1:length(rates)){
  ri = rates[i]
  wi = weights[i]

  blen.50 <- blen.param$blen.50 * ri
  blen.51 <- blen.param$blen.51 * ri
  blen.62 <- blen.param$blen.62 * ri
  blen.63 <- blen.param$blen.63 * ri
  blen.75 <- blen.param$blen.75 * ri
  blen.76 <- blen.param$blen.76 * ri
  blen.87 <- blen.param$blen.87 * ri
  blen.84 <- blen.param$blen.84 * ri

  Ptr.50 <- expm(Q*blen.50)
  Ptr.51 <- expm(Q*blen.51)
  Ptr.62 <- expm(Q*blen.62)
  Ptr.63 <- expm(Q*blen.63)
  Ptr.75 <- expm(Q*blen.75)
  Ptr.76 <- expm(Q*blen.76)
  Ptr.87 <- expm(Q*blen.87)
  Ptr.84 <- expm(Q*blen.84)

  # Now calculate left (#.1) and right (#.2) post-order (conditional likelihood) of each internal node
  post.5.left <- Ptr.50 %*% post.0
  post.5.right <- Ptr.51 %*% post.1
  post.5 <- post.5.left * post.5.right
  post.6.left <- Ptr.62 %*% post.2
  post.6.right <- Ptr.63 %*% post.3
  post.6 <- post.6.left * post.6.right
  post.7.left <- Ptr.75 %*% post.5
  post.7.right <- Ptr.76 %*% post.6
  post.7 <- post.7.left * post.7.right
  post.8.left <- Ptr.87 %*% post.7
  post.8.right <- Ptr.84 %*% post.4
  post.8 <- post.8.left * post.8.right

  likelihood.mat <- rbind(likelihood.mat, wi*colSums(stationary.dist * post.8))

  # Now calculate pre-order partials
  node.7.pre.order <- crossprod(Ptr.87, node.8.pre.order * post.8.right)
  node.4.pre.order <- crossprod(Ptr.84, node.8.pre.order * post.8.left)
  node.5.pre.order <- crossprod(Ptr.75, node.7.pre.order * post.7.right)
  node.6.pre.order <- crossprod(Ptr.76, node.7.pre.order * post.7.left)
  node.0.pre.order <- crossprod(Ptr.50, node.5.pre.order * post.5.right)
  node.1.pre.order <- crossprod(Ptr.51, node.5.pre.order * post.5.left)
  node.2.pre.order <- crossprod(Ptr.62, node.6.pre.order * post.6.right)
  node.3.pre.order <- crossprod(Ptr.63, node.6.pre.order * post.6.left)

  node.0.pre.order.list[[i]] <- t(as.matrix(node.0.pre.order))
  node.1.pre.order.list[[i]] <- t(as.matrix(node.1.pre.order))
  node.2.pre.order.list[[i]] <- t(as.matrix(node.2.pre.order))
  node.3.pre.order.list[[i]] <- t(as.matrix(node.3.pre.order))
  node.4.pre.order.list[[i]] <- t(as.matrix(node.4.pre.order))
  node.5.pre.order.list[[i]] <- t(as.matrix(node.5.pre.order))
  node.6.pre.order.list[[i]] <- t(as.matrix(node.6.pre.order))
  node.7.pre.order.list[[i]] <- t(as.matrix(node.7.pre.order))

  node.0.gradient <- NULL
  node.1.gradient <- NULL
  node.2.gradient <- NULL
  node.3.gradient <- NULL
  node.4.gradient <- NULL
  node.5.gradient <- NULL
  node.6.gradient <- NULL
  node.7.gradient <- NULL
  for(j in 1:dim(data.0)[2]){
    node.0.gradient <- c(node.0.gradient, post.0[, j] %*% t(Q.normalized) %*% node.0.pre.order[, j] / sum(post.0[, j] * node.0.pre.order[, j]))
    node.1.gradient <- c(node.1.gradient, post.1[, j] %*% t(Q.normalized) %*% node.1.pre.order[, j] / sum(post.1[, j] * node.1.pre.order[, j]))
    node.2.gradient <- c(node.2.gradient, post.2[, j] %*% t(Q.normalized) %*% node.2.pre.order[, j] / sum(post.2[, j] * node.2.pre.order[, j]))
    node.3.gradient <- c(node.3.gradient, post.3[, j] %*% t(Q.normalized) %*% node.3.pre.order[, j] / sum(post.3[, j] * node.3.pre.order[, j]))
    node.4.gradient <- c(node.4.gradient, post.4[, j] %*% t(Q.normalized) %*% node.4.pre.order[, j] / sum(post.4[, j] * node.4.pre.order[, j]))
    node.5.gradient <- c(node.5.gradient, post.5[, j] %*% t(Q.normalized) %*% node.5.pre.order[, j] / sum(post.5[, j] * node.5.pre.order[, j]))
    node.6.gradient <- c(node.6.gradient, post.6[, j] %*% t(Q.normalized) %*% node.6.pre.order[, j] / sum(post.6[, j] * node.6.pre.order[, j]))
    node.7.gradient <- c(node.7.gradient, post.7[, j] %*% t(Q.normalized) %*% node.7.pre.order[, j] / sum(post.7[, j] * node.7.pre.order[, j]))
  }
  node.0.gradient.mat <- rbind(node.0.gradient.mat, node.0.gradient)
  node.1.gradient.mat <- rbind(node.1.gradient.mat, node.1.gradient)
  node.2.gradient.mat <- rbind(node.2.gradient.mat, node.2.gradient)
  node.3.gradient.mat <- rbind(node.3.gradient.mat, node.3.gradient)
  node.4.gradient.mat <- rbind(node.4.gradient.mat, node.4.gradient)
  node.5.gradient.mat <- rbind(node.5.gradient.mat, node.5.gradient)
  node.6.gradient.mat <- rbind(node.6.gradient.mat, node.6.gradient)
  node.7.gradient.mat <- rbind(node.7.gradient.mat, node.7.gradient)
}
node.0.gradient <- sum(colSums(node.0.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.50
node.1.gradient <- sum(colSums(node.1.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.51
node.2.gradient <- sum(colSums(node.2.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.62
node.3.gradient <- sum(colSums(node.3.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.63
node.4.gradient <- sum(colSums(node.4.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.84
node.5.gradient <- sum(colSums(node.5.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.75
node.6.gradient <- sum(colSums(node.6.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.76
node.7.gradient <- sum(colSums(node.7.gradient.mat * (weights * rates * likelihood.mat)) / colSums(weights * likelihood.mat)) * blen.param$blen.87

ll <- getLoglikelihood(data, rate.param, blen.param, stationary.dist, rates, weights)
cat("logL = ", formatC(signif(ll,digits=18), digits=16,format="fg", flag="#"))
cat("logL = ", formatC(signif(sum(log(colSums(likelihood.mat))),digits=18), digits=16,format="fg", flag="#"))

print(c(node.0.gradient, node.1.gradient, node.2.gradient, node.3.gradient, node.4.gradient, node.5.gradient, node.6.gradient, node.7.gradient))
