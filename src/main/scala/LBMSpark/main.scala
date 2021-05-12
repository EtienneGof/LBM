package LBMSpark

import Common.DataGeneration.{randomLBMDataGeneration, simulatedDataDenseMatrixToRDD}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object main {

  implicit val ss: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("AnalysePlan")
    .config("spark.executor.cores", 2)
    //.config("spark.executor.memory", "30G")
    .config("spark.executor.heartbeatInterval", "20s")
    .config("spark.driver.memory", "10G")
    .getOrCreate()

  ss.sparkContext.setLogLevel("WARN")
  ss.sparkContext.setCheckpointDir("checkpointDir")
  val confSpark: SparkConf = new SparkConf().setMaster("local[2]").setAppName("LBM")

  def main(args: Array[String]) {


    implicit val ss: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("AnalysePlan")
      .config("spark.executor.cores", 2)
      //.config("spark.executor.memory", "30G")
      .config("spark.executor.heartbeatInterval", "20s")
      .config("spark.driver.memory", "10G")
      .getOrCreate()

    val K = 2
    val L = 2

    val multivariateFullCovarianceMatrices: List[List[DenseMatrix[Double]]] = (0 until L).map(l => {
      (0 until K).map(k => {
        DenseMatrix.tabulate[Double](3,3){ (s,t) => if(s == t) {(k + l + 1) * 0.5} else {(k + l) * 0.2}}
      }).toList
    }).toList

    val multivariateModes: List[List[DenseVector[Double]]] = List(List(DenseVector(-2D,-2D,-2D),
      DenseVector(-1D,1D,-1D)),List(DenseVector(5D,5D,0D), DenseVector(10D,10D,0D)))
    val multivariateSizeRowPartition = List(30,70)
    val multivariateSizeColPartition = List(25,50)

    val multivariateDataRDD: RDD[(Int, Array[DenseVector[Double]])] = simulatedDataDenseMatrixToRDD(
      randomLBMDataGeneration(
        multivariateModes,
        multivariateFullCovarianceMatrices,
        multivariateSizeRowPartition,
        multivariateSizeColPartition))

    val sem = new LBMSpark.SEM(multivariateDataRDD, K, L)
    sem.run(5)

  }
}
