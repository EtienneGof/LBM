package Common

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Matrices, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Tools extends java.io.Serializable {

  def prettyPrint(sizePerBlock: Map[(Int,Int), Int]): Unit = {

    val keys = sizePerBlock.keys
    val L = keys.map(_._1).max + 1
    val K = keys.map(_._2).max + 1
    val mat = DenseMatrix.tabulate[String](L,K){
      case (i, j) =>
        if(sizePerBlock.contains((i,j))){
          sizePerBlock(i,j).toString
        } else {"-"}
    }

    println(mat.t)

  }

  def sortedCount(list: List[Int]): List[Int] = {
    list.groupBy(identity).mapValues(_.size).toList.sortBy(_._1).map(_._2)
  }

  def sortedFrequency(list: List[Int]): List[Double] = {
    sortedCount(list).map(_ / list.length.toDouble)
  }

  def argmax(l: List[Double]): Int ={
    l.view.zipWithIndex.maxBy(_._1)._2
  }

  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) : Traversable[(X,Y)] = for { x <- xs; y <- ys } yield (x, y)
  }

  def denseMatrixToMatrix(A: DenseMatrix[Double]): Matrix = {
    Matrices.dense(A.rows, A.cols, A.toArray)
  }

  def joinLBMRowPartitionToData(data: RDD[(Int, Array[DenseVector[Double]])],
                                rowPartition: List[Int],
                                n:Int)(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], Int)] = {
    val rowPartitionPerRow: List[(Int,Int)] =
      (0 until n).map(i => (i, rowPartition(i))).toList
    data.join(ss.sparkContext.parallelize(rowPartitionPerRow, 30))
      .map(r => {(r._1, r._2._1, r._2._2)})
  }


  def getSizeAndSumByBlock(data: RDD[((Int, Int), Array[DenseVector[Double]])]): RDD[((Int, Int), (DenseVector[Double], Int))] = {
    data
      .map(r => (r._1, (r._2.reduce(_+_), r._2.length)))
      .reduceByKey((a,b) => (a._1 + b._1, a._2 + b._2))
  }

  private def getDataPerColumnAndRow(periodogram: RDD[(Int, Array[DenseVector[Double]], Int)],
                                     partitionPerColBc: Broadcast[DenseVector[Int]],
                                     L:Int) = {

    val dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])] = periodogram.flatMap(row => {
      (0 until L).map(l => {
        val rowDv = DenseVector(row._2)
        val cells = rowDv(partitionPerColBc.value :== l)
        ((l, row._3), cells.toArray)
      })
    }).cache()
    dataPerColumnAndRow
  }

  private def getCovarianceMatrices(dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])],
                                    meanByBlock: Map[(Int, Int), DenseVector[Double]],
                                    sizeBlockMap: Map[(Int, Int), Int],
                                    L:Int,
                                    K:Int) = {

    val sumCentered = dataPerColumnAndRow.map(r => {
      (r._1, r._2.map(v => {
        val centeredValue = v - meanByBlock(r._1)
        centeredValue * centeredValue.t
      }).reduce(_ + _))
    }).reduceByKey(_ + _).collect().toMap
    val res = (0 until L).map(l => {
      (0 until K).map(k => {
        sumCentered(l, k) / (sizeBlockMap(l, k).toDouble - 1)
      }).toList
    }).toList
    res

  }

  def getMeansAndCovariances(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                             colPartition: Broadcast[DenseVector[Int]],
                             L:Int,
                             K:Int): (List[List[DenseVector[Double]]], List[List[DenseMatrix[Double]]], RDD[((Int, Int), Int)], Map[(Int, Int), Int]) = {
    val dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])] = getDataPerColumnAndRow(data, colPartition, L)
    val sizeAndSumBlock = getSizeAndSumByBlock(dataPerColumnAndRow)
    val sizeBlock = sizeAndSumBlock.map(r => (r._1, r._2._2))
    val sizeBlockMap = sizeBlock.collect().toMap
    Common.Tools.prettyPrint(sizeBlockMap)
    val meanByBlock: Map[(Int, Int), DenseVector[Double]] = sizeAndSumBlock.map(r => (r._1, r._2._1 / r._2._2.toDouble)).collect().toMap
    val covMat = getCovarianceMatrices(dataPerColumnAndRow, meanByBlock, sizeBlockMap, L, K)
    val listMeans = (0 until L).map(l => {
      (0 until K).map(k => {
        meanByBlock(l, k)
      }).toList
    }).toList
    (listMeans, covMat, sizeBlock, sizeBlockMap)
  }

}
