package Common

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.MultivariateGaussian
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import Common.ProbabilisticTools._
import scala.util.Random

object DataGeneration  {

  def simulatedDataDenseMatrixToRDD(data: DenseMatrix[DenseVector[Double]], numPartition:Int = 0)(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]])] ={
    val dataList: Seq[(Int, Array[DenseVector[Double]])] = (0 until data.rows).map(i => {
      (i, data(i,::).t.toArray)
    })
    if(numPartition == 0){
      ss.sparkContext.parallelize(dataList)
    } else {
      ss.sparkContext.parallelize(dataList,numPartition)
    }
  }

  def randomLBMDataGeneration(modes: List[List[DenseVector[Double]]],
                              covariances: List[List[DenseMatrix[Double]]],
                              sizeClusterRow: List[Int],
                              sizeClusterCol: List[Int],
                              shuffle: Boolean = false,
                              mixingProportion: Double = 0D): DenseMatrix[DenseVector[Double]] = {

    val modeLengths = modes.map(_.length)
    val K = modeLengths.head
    val covLengths = covariances.map(_.length)
    val L = modes.length

    require(modeLengths.forall(_ == modeLengths.head), "In LBM case, every column must have the same number of modes")
    require(covLengths.forall(_ == covLengths.head), "In LBM case, every column must have the same number of covariances matrices")
    require(K == covLengths.head, "modes and covariances K do not match")
    require(modes.length == covariances.length, "modes and covariances L do not match")
    require(sizeClusterRow.length == K)
    require(sizeClusterCol.length == L)

    val sizeClusterRowEachColumn = List.fill(L)(sizeClusterRow)
    val dataPerBlock: List[List[DenseMatrix[DenseVector[Double]]]] = generateDataPerBlock(
      modes,
      covariances,
      sizeClusterRowEachColumn,
      sizeClusterCol,
      mixingProportion)
    val data: DenseMatrix[DenseVector[Double]] = nestedReduce(dataPerBlock)

    applyConditionalShuffleData(data, shuffle)

  }

  def generateDataPerBlock(modes: List[List[DenseVector[Double]]],
                           covariances: List[List[DenseMatrix[Double]]],
                           sizeClusterRow: List[List[Int]],
                           sizeClusterCol: List[Int],
                           mixingProportion: Double=0D): List[List[DenseMatrix[DenseVector[Double]]]]={

    require(mixingProportion>=0D & mixingProportion <=1D)
    val L = modes.length
    val KVec = modes.map(_.length)
    val blockDistributions = (0 until L).map(l => {
      modes(l).indices.map(k => {
        MultivariateGaussian(modes(l)(k),covariances(l)(k))
      })
    })
    modes.indices.map(l => {
      modes(l).indices.map(k => {
        val dataList: Array[DenseVector[Double]] = blockDistributions(l)(k).sample(sizeClusterRow(l)(k) * sizeClusterCol(l)).toArray
        val mixingIndicator = dataList.indices.map(_ => sample(List(1 - mixingProportion, mixingProportion))).toList

        val mixedData = (dataList zip mixingIndicator).map(c =>
          if(c._2==1){
            val newColumnCluster = (l+1)%L
            val newRowCluster = (k+1)%KVec(newColumnCluster)
            blockDistributions(newColumnCluster)(newRowCluster).draw()
          } else {
            c._1
          })
        DenseMatrix(mixedData).reshape(sizeClusterRow(l)(k),sizeClusterCol(l))
      }).toList
    }).toList
  }

  def nestedReduce(dataList: List[List[DenseMatrix[DenseVector[Double]]]]): DenseMatrix[DenseVector[Double]] = {
    dataList.indices.map(l => {
      dataList(l).indices.map(k_l => {
        dataList(l)(k_l)
      }).reduce(DenseMatrix.vertcat(_,_))
    }).reduce(DenseMatrix.horzcat(_,_))
  }

  def applyConditionalShuffleData(data: DenseMatrix[DenseVector[Double]], shuffle:Boolean): DenseMatrix[DenseVector[Double]]= {
    if(shuffle){
      val newRowIndex: List[Int] = Random.shuffle((0 until data.rows).toList)
      val newColIndex: List[Int] = Random.shuffle((0 until data.cols).toList)
      DenseMatrix.tabulate[DenseVector[Double]](data.rows,data.cols){ (i,j) => data(newRowIndex(i), newColIndex(j))}
    } else data
  }

}
