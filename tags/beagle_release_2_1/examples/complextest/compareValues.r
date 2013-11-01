evec = matrix(
  c(			 -0.5,  0.6906786606674509,   0.15153543380548623, 0.5,
			  0.5, -0.15153543380548576,  0.6906786606674498,  0.5,
			 -0.5, -0.6906786606674498,  -0.15153543380548617, 0.5,
			  0.5,  0.15153543380548554, -0.6906786606674503,  0.5),
  nrow=4,ncol=4,byrow=T)

ievc = matrix(
  c(			 -0.5,  0.5, -0.5,  0.5,
			  0.6906786606674505, -0.15153543380548617, -0.6906786606674507,   0.15153543380548645,
			  0.15153543380548568, 0.6906786606674509,  -0.15153543380548584, -0.6906786606674509,
			  0.5,  0.5,  0.5,  0.5),
  nrow=4,ncol=4,byrow=T)

make.expDt = function(t) {
  matrix(
  c(exp(-2*t),                   0,                   0,  0,
            0,  exp(-1*t)*cos(1*t),  exp(-1*t)*sin(1*t),  0,
            0, -exp(-1*t)*sin(1*t),  exp(-1*t)*cos(1*t),  0,
            0,                   0,                   0,  1),
  nrow=4,ncol=4,byrow=T)
}

q.circulant = matrix(
  c(-1,  1,  0,  0,
     0, -1,  1,  0,
     0,  0, -1,  1,
     1,  0,  0, -1),
  nrow=4,ncol=4,byrow=T)

test.1 = evec %*% make.expDt(0.1) %*% ievc # Schur decomposition (method used in BEAGLE)

Evec = eigen(q.circulant)$vectors # Complex eigendecomposition
Ievc = solve(Evec)
make2.expDt = function(t) {
  diag(exp(eigen(q.circulant)$values * t))
}

test.2 = Evec %*% make2.expDt(0.1) %*% Ievc  # Should (and does) equal test.1

out.1 = t(make.expDt(0.1)) %*% t(evec) # Should be intermediate result

out.2 = t(ievc) %*% out.1 # Should (and does) equal t(test.1)




