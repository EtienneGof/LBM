package LBMSpark

import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax}
import breeze.numerics.{exp, log}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.{Failure, Random, Success, Try}


class SEM(data: RDD[(Int, Array[DenseVector[Double]])],
          K: Int,
          L: Int,
          providedInitialRowPartition: Option[List[Int]] = None,
          providedInitialColPartition: Option[List[Int]] = None)(implicit ss: SparkSession) extends Serializable {

  val n: Int = data.count().toInt
  val p: Int = data.first()._2.length
  val precision = 1e-5

  val rowPartition: List[Int] = providedInitialRowPartition match {
    case Some(rp) =>
      require(rp.distinct.length == K)
      require(rp.length == n)
      rp
    case None => Random.shuffle((0 until n).map(i => i % K)).toList
  }
  var dataWithRowPartition: RDD[(Int, Array[DenseVector[Double]], Int)] = Common.Tools.joinLBMRowPartitionToData(data, rowPartition, n)

  var colPartition: List[Int] = providedInitialColPartition match {
    case Some(cp) =>
      require(cp.distinct.length == L)
      require(cp.length == p)
      cp
    case None => Random.shuffle((0 until p).map(i => i%L)).toList
  }

  var rowProportions: List[Double] = sortedFrequency(rowPartition)
  var colProportions: List[Double] = sortedFrequency(colPartition)
  var components: List[List[MultivariateGaussian]] = getComponentDistributions._1

  def computeJointLogDistribColsFromSample()(implicit ss: SparkSession): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(colProportions.map(log(_)).toArray)
    val gaussianBc = ss.sparkContext.broadcast(components)

    val D: RDD[DenseMatrix[Double]] =
      dataWithRowPartition.map(row => {
        row._2.indices.map(j => {
          DenseMatrix((0 until this.L).map(l => {
            gaussianBc.value(l)(row._3).logpdf(Vectors.dense(row._2(j).toArray))
          }).toArray)
        }).reduce((a, b) => DenseMatrix.vertcat(a, b))
      })

    val prob = D.reduce(_ + _)
    val sumProb = prob(*, ::).map(dv => dv.toArray.toList).toArray.toList.zipWithIndex.map(e =>
      (DenseVector(e._1.toArray) + logPiCols).toArray.toList)
    sumProb
  }

  def drawColPartition()(implicit ss: SparkSession): List[Int] = {
    val jointLogDistribCols: List[List[Double]] = computeJointLogDistribColsFromSample()
    jointLogDistribCols.map(x => {
      val LSE = logSumExp(x)
      sample(x.map(e => exp(e - LSE)))
    })
  }

  def drawRowPartition()(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], Int)] = {

    val logPiBc = ss.sparkContext.broadcast(DenseVector(rowProportions.map(log(_)).toArray))
    val colPartBc = ss.sparkContext.broadcast(colPartition)
    val gaussianBc = ss.sparkContext.broadcast(components)
    val resRDD = data.map(row => {
      val prob = (0 until this.K).map(k => {
        row._2.indices.map(j => {
          gaussianBc.value(colPartBc.value(j))(k).logpdf(Vectors.dense(row._2(j).toArray))
        }).sum + logPiBc.value(k)
      }).toList
      val LSE = logSumExp(prob)

      (row._1,
        row._2,
        sample(prob.map(e => exp(e - LSE))))
    })
    resRDD
  }

  def expectationStep(nIter: Int = 3,
                      verbose: Boolean = true)(implicit ss: SparkSession): (RDD[(Int, Array[DenseVector[Double]], Int)], List[Int]) = {

    dataWithRowPartition = drawRowPartition()
    colPartition = drawColPartition()
    var k: Int = 0
    while (k < nIter) {
      dataWithRowPartition = drawRowPartition()
      colPartition = drawColPartition()
      k += 1
    }
    (dataWithRowPartition, colPartition)
  }

  def getComponentDistributions: (List[List[MultivariateGaussian]], RDD[((Int,Int), Int)]) = {

    val partitionPerColBc = ss.sparkContext.broadcast(DenseVector(colPartition:_*))
    val (means, covMat, sizeBlock, _) = getMeansAndCovariances(dataWithRowPartition, partitionPerColBc, L, K)
//    partitionPerColBc.destroy()

    val componentDistributions = (0 until L).map(l => {
      (0 until K).map(k => {
        new MultivariateGaussian(
          Vectors.dense(means(l)(k).toArray),
          denseMatrixToMatrix(covMat(l)(k)))
      }).toList
    }).toList

    (componentDistributions, sizeBlock)
  }

  def maximizationStep(implicit ss: SparkSession): Unit = {
    println("maximization step")
    val componentsAndSizeBlocks = getComponentDistributions
    components = componentsAndSizeBlocks._1
    estimateProportionFromLBMSizeBlock(componentsAndSizeBlocks._2)
  }

  def estimateProportionFromLBMSizeBlock(sizeBlock: RDD[((Int, Int), Int)]): Unit = {
    val sizeCol = sizeBlock.map(r => (r._1._1, r._2)).reduceByKey(_ + _).collect().sortBy(_._1).map(_._2).toList
    val sizeRow = sizeBlock.map(r => (r._1._2, r._2)).reduceByKey(_ + _).collect().sortBy(_._1).map(_._2).toList

    val p = sizeCol.sum
    val n = sizeRow.sum

    colProportions = sizeCol.map(_/p.toDouble)
    rowProportions = sizeRow.map(_/n.toDouble)
  }

  def completeLogLikelihood(): Double = {

    val logRho: List[Double] = colProportions.map(log(_))
    val logPi: List[Double]  = rowProportions.map(log(_))

    dataWithRowPartition.map(row => {
      row._2.indices.map(j => {
        val l = colPartition(j)
        logPi(row._3)
        + logRho(l)
        + components(colPartition(j))(row._3).logpdf(Vectors.dense(row._2(j).toArray))
      }).sum
    }).sum
  }

  def ICL(): Double = {
    val dimVar = components.head.head.mean.size
    val nParamPerComponent =  dimVar + dimVar * (dimVar + 1) / 2D
    completeLogLikelihood() - log(n) * (K - 1) / 2D - log(p) * (L - 1) / 2D - log(n * p) * (K * L * nParamPerComponent) / 2D
  }

  def run(maxIterations:Int = 10, verbose:Boolean=false): (List[Int], List[Int], Double, Double) = {

    var iter = 0
    do {
      iter += 1
      expectationStep(verbose = verbose)
      maximizationStep
    } while (iter < maxIterations)

    (rowPartition, colPartition, completeLogLikelihood(), ICL())
  }

  def concurrentRun(nConcurrent: Int,
                    maxIterations:Int = 10,
                    verbose:Boolean=false): (List[Int], List[Int], Double, Double) = {

    val allRes = (0 until nConcurrent).map(_ => {
      val sem = new LBMSpark.SEM(data, K, L)
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
