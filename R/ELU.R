#'@importFrom plyr aaply
ELU <- function(x, data, fun, dfun, prior, dprior, tol) {
  fun.results <- unname(aaply(data, 1, function(d) {
    fun(params = x, x = d)
  }))

  fun.gradients <- unname(aaply(data, 1, function(d) {
    dfun(params = x, x = d)
  }))

  density <- prior(x)
  density.gradient <- dprior(x)

  el <- emplik::el.test(fun.results, rep(0, ncol(fun.results)))

  if(abs(mean(el$wts) - 1) > tol) {
    CustomStop("invalid_el_weight_mean",
                "Mean of EL weights is not close enough to 1.")
  }

  u <- - sum(log(density)) - sum(log(el$wts / length(el$wts)))

  dellogL <- array(0, c(length(el$wts), length(x)))
  for (i in 1:NROW(data))
  {
    dellogL[i, ] <- el$wts[i] * t(as.matrix(el$lambda)) %*% fun.gradients[i, , ]
  }
  gradient <- unname(aaply(dellogL, 2, sum)) - density.gradient

  result <- u
  attr(result, "gradient") <- gradient

  return(result)
}
