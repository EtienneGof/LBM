package LBM

import Common.Tools._
import Common.ProbabilisticTools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}
import breeze.numerics.{abs, exp, log}
import breeze.stats.distributions.MultivariateGaussian

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class SEM(data: DenseMatrix[DenseVector[Double]],
          K: Int,
          L: Int,
          providedInitialRowPartition: Option[List[Int]] = None,
          providedInitialColPartition: Option[List[Int]] = None) extends Serializable {

  val n: Int = data.rows
  val p: Int = data.cols
  val dataList: List[DenseVector[Double]] = data.t.toArray.toList

  val precision = 1e-5

  var rowPartition: List[Int] = providedInitialRowPartition match {
    case Some(rp) =>
      require(rp.distinct.length == K)
      require(rp.length == n)
      rp
    case None => Random.shuffle((0 until n).map(i => i%K)).toList
  }

  val colPartition: List[Int] = providedInitialColPartition match {
    case Some(cp) =>
      require(cp.distinct.length == L)
      require(cp.length == p)
      cp
    case None => Random.shuffle((0 until p).map(i => i % L)).toList
  }

  var rowProportions: List[Double] = sortedFrequency(rowPartition)
  var colProportions: List[Double] = sortedFrequency(colPartition)
  var components: List[List[MultivariateGaussian]] = getComponentDistributions

  def computeJointLogDistribRowsFromSample(): List[List[Double]] = {

    // Vector of log p(z=k ; theta)
    val logPiRows: List[Double] = rowProportions.map(log(_))
    val logPdfPerRowComponentMat: List[List[Double]] = (0 until n).par.map(idxRow => {
      (0 until K).map(k => {
        val f_k = (0 until p).map(idxCol => {
          components(colPartition(idxCol))(k).logPdf(data(idxRow,idxCol))
        }).sum
        logPiRows(k) + f_k
      }).toList
    }).toList
    logPdfPerRowComponentMat
  }

  def computeJointLogDistribColsFromSample(): List[List[Double]] = {

    // Vector of log p(z=k ; theta)
    val logPiCols: List[Double] = colProportions.map(log(_))

    val logPdfPerRowComponentMat: List[List[Double]] = (0 until p).par.map(idxCol => {
      (0 until L).map(l => {
        val f_l = (0 until n).map(idxRow => {
          components(l)(rowPartition(idxRow)).logPdf(data(idxRow,idxCol))
        }).sum
        logPiCols(l) + f_l
      }).toList
    }).toList

    logPdfPerRowComponentMat
  }

  def drawPartition(jointLogDistrib: List[List[Double]]): List[Int] = {
    jointLogDistrib.par.map(x => {
      val LSE = logSumExp(x)
      sample(x. map(e => exp(e - LSE)))
    }).toList
  }

  def expectationStep(nIter: Int = 3,
                      verbose: Boolean = false): Unit = {

    @tailrec def updatePartitions(iter: Int): Unit = {
      if (iter <= nIter){
        rowPartition = drawPartition(computeJointLogDistribRowsFromSample())
        rowPartition = drawPartition(computeJointLogDistribColsFromSample())
        updatePartitions(iter + 1)
      }
    }

    updatePartitions(1)

    if(verbose){
      println("End Stochastic-Expectation step")
    }

  }

  def getComponentDistributions: List[List[MultivariateGaussian]] = {

    val blockPartition = (rowPartition cross colPartition).toList
    val dataPartition = dataList zip blockPartition

    (0 until L).par.map(l => {
      (l,
        (0 until K).par.map(k => {
          val filteredData: List[DenseVector[Double]] = dataPartition.filter(_._2==(k,l)).map(_._1).toArray.toList
          val sizeBlock: Int = filteredData.length
          val mode:DenseVector[Double] = filteredData.reduce(_+_) / sizeBlock.toDouble
          val covMat: DenseMatrix[Double] = filteredData.map(v => {
            val centeredRow: DenseVector[Double] = v - mode
            centeredRow * centeredRow.t}).reduce(_+_) / sizeBlock.toDouble
          (k, MultivariateGaussian(mode, covMat))
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def maximizationStep(verbose:Boolean = false): Unit = {

    rowProportions = sortedFrequency(rowPartition)
    colProportions = sortedFrequency(colPartition)
    components = getComponentDistributions

    if(verbose){println("End Maximization")}

  }

  def completeLogLikelihood(): Double = {

    val logRho: List[Double] = colProportions.map(log(_))
    val logPi: List[Double]  = rowProportions.map(log(_))

    (0 until p).par.map(j => {
      (0 until n).map(i => {
        val k = rowPartition(i)
        val l = colPartition(j)
        logPi(k)
        + logRho(l)
        + components(l)(k).logPdf(data(i,j))
      }).sum
    }).sum

  }

  def ICL(): Double = {
    val dimVar = components.head.head.mean.size
    val nParamPerComponent = dimVar+ dimVar*(dimVar+1)/2D
    completeLogLikelihood() - log(n) * (this.K - 1) / 2D - log(p) * (L - 1)/2D - log(n * p) * (K * L * nParamPerComponent) / 2D
  }

  def run(maxIterations:Int = 10, verbose:Boolean=false): (List[Int], List[Int], Double, Double) = {

    var iter = 0
    do {
      iter += 1
      expectationStep(verbose = verbose)
      maximizationStep(verbose = verbose)
    } while (iter < maxIterations)

    (rowPartition, colPartition, completeLogLikelihood(), ICL())
  }

  def concurrentRun(nConcurrent: Int, maxIterations:Int = 10, verbose:Boolean=false): (List[Int], List[Int], Double, Double) = {

    val allRes = (0 until nConcurrent).map(_ => {
      val sem = new SEM(data, K, L)
      Try( sem.run(maxIterations) ) match {
        case Success(v) =>
          Success(v).get
        case Failure(_) =>
          (rowPartition, colPartition, Double.NegativeInfinity, Double.NegativeInfinity)
      }
    })
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_._3).toArray)
    allRes(argmax(allLikelihoods))
  }

}
