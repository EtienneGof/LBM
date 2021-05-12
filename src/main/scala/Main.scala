import Common.DataGeneration
import Common.DataGeneration.{randomLBMDataGeneration, simulatedDataDenseMatrixToRDD}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Main {

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

  def main(args: Array[String]): Unit = {

    val modes = List(List(DenseVector(10D, 10D),DenseVector(10D, -10)),
      List(DenseVector(-10D, 10),DenseVector(-10D, -10)))

    val covariances = List(List(
      DenseMatrix(0.05, 0.0, 0.0, 0.05).reshape(2,2),
      DenseMatrix(0.07, 0.0, 0.0, 0.07).reshape(2,2)),
      List(DenseMatrix(0.02, 0.0, 0.0, 0.02).reshape(2,2),
      DenseMatrix(0.03, 0.0, 0.0, 0.03).reshape(2,2)))

    val trueClusterRowSize = List(4,6)
    val trueClusterColSize = List(6,8)

    val data = DataGeneration.randomLBMDataGeneration(modes, covariances, trueClusterRowSize, trueClusterColSize)


    val sem = new LBM.SEM(data, K = 2, L = 2)
    val results = sem.run()

    println("####################")

    val dataRDD: RDD[(Int, Array[DenseVector[Double]])] = simulatedDataDenseMatrixToRDD(
      randomLBMDataGeneration(
        modes,
        covariances,
        trueClusterRowSize,
        trueClusterColSize))

    val semSpark = new LBMSpark.SEM(dataRDD, K = 2, L = 2)
    semSpark.run(5)

  }
}
