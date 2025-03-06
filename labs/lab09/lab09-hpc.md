Lab 9 - HPC
================

# Learning goals

In this lab, you are expected to practice the following skills:

- Evaluate whether a problem can be parallelized or not.
- Practice with the parallel package.
- Use Rscript to submit jobs.

## Problem 1

Give yourself a few minutes to think about what you learned about
parallelization. List three examples of problems that you believe may be
solved using parallel computing, and check for packages on the HPC CRAN
task view that may be related to it.

Monte Carlo Simulations - Many iterations of random sampling can be
parallelized. Related package: parallel, foreach.

Genome-Wide Association Studies (GWAS) - Large-scale computations on
genetic data can be parallelized. Related package: BiocParallel.

Image Processing (e.g., edge detection, filtering) - Parallel processing
of image pixels can speed up performance. Related package: future.apply.

## Problem 2: Pre-parallelization

The following functions can be written to be more efficient without
using `parallel`:

1.  This function generates a `n x k` dataset with all its entries
    having a Poisson distribution with mean `lambda`.

``` r
fun1 <- function(n = 100, k = 4, lambda = 4) {
  x <- NULL
  
  for (i in 1:n)
    x <- rbind(x, rpois(k, lambda))
  
  return(x)
}

fun1alt <- function(n = 100, k = 4, lambda = 4) {
 matrix(rpois(n * k, lambda), nrow = n, ncol = k)
}

# Benchmarking
microbenchmark::microbenchmark(
  fun1(),
  fun1alt()
)
```

    ## Warning in microbenchmark::microbenchmark(fun1(), fun1alt()): less accurate
    ## nanosecond times to avoid potential integer overflows

    ## Unit: microseconds
    ##       expr     min      lq      mean   median       uq      max neval
    ##     fun1() 101.516 105.821 177.63619 109.1625 112.0120 6770.084   100
    ##  fun1alt()   8.856   9.512  16.18557   9.7785  10.3935  616.394   100

How much faster?

Around 127 microseconds faster (on average)

2.  Find the column max (hint: Checkout the function `max.col()`).

``` r
# Data Generating Process (10 x 10,000 matrix)
set.seed(1234)
x <- matrix(rnorm(1e4), nrow=10)

# Find each column's max value
fun2 <- function(x) {
  apply(x, 2, max)
}

fun2alt <- function(x) {
  x[max.col(t(x), ties.method = "first") + (0:(ncol(x)-1))*nrow(x)]
}

# Benchmarking
library(ggplot2)
library(microbenchmark)

benchmark_results <- microbenchmark(
  fun2(x),
  fun2alt(x)
)

# Plotting the benchmark results
plot(benchmark_results)
```

![](lab09-hpc_files/figure-gfm/p2-fun2-1.png)<!-- -->

*Answer here with a plot.*

## Problem 3: Parallelize everything

We will now turn our attention to non-parametric
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
Among its many uses, non-parametric bootstrapping allow us to obtain
confidence intervals for parameter estimates without relying on
parametric assumptions.

The main assumption is that we can approximate many experiments by
resampling observations from our original dataset, which reflects the
population.

This function implements the non-parametric bootstrap:

``` r
  library(parallel)

my_boot <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)

  # Making the cluster using `ncpus`
  cl <- makeCluster(ncpus)
  clusterExport(cl, varlist = c("dat", "stat", "idx"), envir = environment())
  
  # Running parallel bootstrapping
  ans <- parLapply(cl, seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  stopCluster(cl)
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  ans
}
```

1.  Use the previous pseudocode, and make it work with `parallel`. Here
    is just an example for you to try:

``` r
# Bootstrap of a linear regression model
my_stat <- function(dat) {
  coef(lm(y ~ x, data = dat))
}

# DATA SIM
set.seed(1)
n <- 500
R <- 1000
x <- rnorm(n)
y <- x*5 + rnorm(n)
dat <- data.frame(x, y)

# Check if we get something similar as lm
ans0 <- confint(lm(y ~ x, data = dat))
ans1 <- my_boot(dat, my_stat, R, ncpus = 2)

# Checking bootstrap results
apply(ans1, 2, quantile, probs = c(0.025, 0.975))
```

    ##       (Intercept)        x
    ## 2.5%  -0.14307029 4.868525
    ## 97.5%  0.05292241 5.048437

2.  Check whether your version actually goes faster than the
    non-parallel version:

``` r
microbenchmark::microbenchmark(
  my_boot(dat, my_stat, R, ncpus = 1),
  my_boot(dat, my_stat, R, ncpus = 2)
)
```

    ## Unit: milliseconds
    ##                                 expr      min       lq     mean   median
    ##  my_boot(dat, my_stat, R, ncpus = 1) 312.2817 324.8945 326.5579 327.0036
    ##  my_boot(dat, my_stat, R, ncpus = 2) 222.0074 224.3665 225.3759 225.3427
    ##        uq      max neval
    ##  328.8394 333.8461   100
    ##  226.2493 232.5565   100

It takes about 30 milliseconds less, on average.

## Problem 4: Compile this markdown document using Rscript

Once you have saved this Rmd file, try running the following command in
your terminal:

``` bash
Rscript --vanilla -e 'rmarkdown::render("[full-path-to-your-Rmd-file.Rmd]")' &
```

Where `[full-path-to-your-Rmd-file.Rmd]` should be replace with the full
path to your Rmd file… :).
