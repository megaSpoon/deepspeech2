package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic

import breeze.math.Complex
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable

class DeepSpeech2NervanaModelTransformer ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {


  def this() = this(Identifiable.randomUID("DFTSpecgram"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)


  override def transform(dataset: Dataset[_]): DataFrame = {

    val model = DeepSpeech2NervanaModelLoader.loadModel[Float](dataset.sparkSession.sparkContext)
    //val model = new DeepSpeech2NervanaModelLoader[Float](9).model
    //val broadcastModel = ModelBroadcast[Float]().broadcast(dataset.sqlContext.sparkContext, model)
    val outputSchema = transformSchema(dataset.schema)
//    val reScale = udf { (samples: Array[Float]) =>
//      val input = Tensor[Float](Storage(samples), 1, Array(1, 1, 13, 398))
//      val output = model.forward(input).toTensor[Float]
//      Vectors.dense(output.storage().toArray.map(_.toDouble))
//      //output.storage().toArray
//    }
    println(s"model length is ${model.parameters()._1.map(_.nElement()).sum}")
    val height = 13
    val reScale = udf { (samples: mutable.WrappedArray[Double]) =>
      //val localModel = broadcastModel.value()
      val width = samples.size / height
      val input = Tensor[Float](Storage(samples.toArray.map(_.toFloat)), 1, Array(1, 1, height, width))
      //val output = localModel.forward(input).toTensor[Float].transpose(2, 3)
      val output = model.forward(input).toTensor[Float].transpose(2, 3)
//      val output = model.output.toTensor[Double].transpose(2, 3)
      output.storage().toArray.map(_.toDouble)
    }

    dataset.withColumn($(outputCol), reScale(col($(inputCol))))
  }


  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): DeepSpeech2NervanaModelTransformer = defaultCopy(extra)
}


object DeepSpeech2NervanaModelTransformer extends DefaultParamsReadable[DeepSpeech2NervanaModelTransformer] {

  override def load(path: String): DeepSpeech2NervanaModelTransformer = {
    super.load(path)
  }
}
