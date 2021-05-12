package LBM

import Common.DataGeneration
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.{Failure, Success, Try}

object main {
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

    val sem = new SEM(data, K = 2, L = 2)
    val results = sem.run()

  }
}
