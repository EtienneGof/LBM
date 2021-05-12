package LBM

import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ModelSelection {

  def gridSearch(data: DenseMatrix[DenseVector[Double]],
                 rangeRow:List[Int],
                 rangeCol:List[Int],
                 verbose: Boolean = false,
                 nConcurrentEachTest:Int=1)(implicit ss: SparkSession): (List[Int], List[Int], Double, Double) = {

    val gridRange: List[(Int, Int)] = (rangeRow cross rangeCol).toList
    val allRes = gridRange.map(KL => {
      println(">>>>> LBM Grid Search try: (K:"+KL._1.toString+", L:"+KL._2.toString+")")
      new SEM(data, KL._1, KL._2).concurrentRun(nConcurrentEachTest)
    })

    val LogLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_._3).toArray)
    val ICLs: DenseVector[Double] = DenseVector(allRes.map(_._4).toArray)

    if(verbose) {
      println()
      gridRange.indices.foreach(i => {
        println("(" + gridRange(i)._1 + ", " + gridRange(i)._2 + "), LogLikelihood: ", LogLikelihoods(i) + ", ICL: " + ICLs(i))
      })
    }

    allRes(argmax(ICLs))
  }


}
