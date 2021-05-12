# LBM

A toy project that performs Multivariate Gaussian Latent Block Inference by Likelihood optimization with Stochastic Expectation Maximization - Gibbs (SEM) algorithm, in Scala and Spark.
 
### What's inside 

The SEM algorithm is implemented in two ways: pure Scala and Scala/Spark. In both cases, a grid-search model selection script is also provided, based on ICL criterion (== Likelihood penalized by model complexity).

### Quick Setup

The script build.sh is provided to build the scala sources. 

See src/pom.xml file for Scala dependencies.

The Main file launches both methods on a tiny toy dataset.

### How to Use ?

These algorithms address the case of datasets composed of real-valued observations. In order to use it on your data, use the following code:

```

    val data = [your data here as DenseMatrix[DenseVector[Double]]
    val sem = new LBM.SEM(data, K = 2, L = 2)
    val resutlts  = sem.run()

```

The output is the state of the MCMC at each iteration, described by the row membership, the column memberships, the logLikehood and the ICL.
In order to counter the inference sensibility to initialization, it is advised to select the best results among several run (a method runConcurrent is available for this purpose).
