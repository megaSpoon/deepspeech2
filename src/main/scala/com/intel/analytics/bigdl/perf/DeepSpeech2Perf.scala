package com.intel.analytics.bigdl.perf

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Container}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, Table}
import com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic.DeepSpeech2NervanaModelLoader
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object DeepSpeech2Perf {

  val parser = new OptionParser[Deepspeech2PerfParam]("DeepSpeech2 Local Performance Test") {
    head("Performance Test of Local Optimizer")
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Int]('b', "batchSize")
      .text("batchSize, default is 4")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]("utt")
      .text("utt length, default is 200")
      .action((v, p) => p.copy(utt = v))
    opt[Int]("coreNumber")
      .text("coreNumber, default is 4")
      .action((v, p) => p.copy(coreNumber = v))
    opt[Int]("partition")
      .text("partition number, default is 4")
      .action((v, p) => p.copy(partition = v))
    opt[String]('d', "inputdata")
      .text("Input data type. One of constant | random")
      .action((v, p) => p.copy(inputData = v))
      .validate(v =>
        if (v.toLowerCase() == "constant" || v.toLowerCase() == "random") {
          success
        } else {
          failure("Input data type must be one of constant and random")
        }
      )
    opt[Boolean]('f', "inference")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(inference = v))
    opt[Boolean]("optim")
      .text("Optimizer. One of true | false")
      .action((v, p) => p.copy(optim = v))
    help("help").text("Prints this usage text")
  }

  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    parser.parse(args, Deepspeech2PerfParam()).foreach(performance)
  }

  def performance(param: Deepspeech2PerfParam): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    val conf = Engine.createSparkConf()
      .setAppName("DS2 Perf")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init
    RandomGenerator.RNG.setSeed(100)

    val uttLen = param.utt
    val batchSize = param.batchSize
    val input = Tensor[Float](batchSize, 1, 13, uttLen)
    val labels = Tensor[Float](batchSize, uttLen / 3).fill(1.0f)
    val model = (new DeepSpeech2NervanaModelLoader[Float](9)).model
    val warmupIteration = 5

    param.inputData match {
      case "constant" => input.fill(0.01f)
      case "random" => input.rand
    }

    def perfLocal(model: Module[Float], criterion: Criterion[Float],
              input: Tensor[Float], target: Tensor[Float], inference: Boolean): Unit = {

      val subModelNumber = param.coreNumber
      val (weight, grad) = model.getParameters()
      val gradLength = grad.nElement()
      logger.info(s"gradLength = ${gradLength}")

      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model and criterion ...")
        val workingModel = if (i == 1) model else model.cloneModule()
        (workingModel, criterion.cloneCriterion())
      }).toArray

      val workingModelWAndG = workingModels.map(_._1.getParameters())
      workingModelWAndG.foreach(_._1.storage().set(weight.storage()))

      val default = Engine.default

      var timeCost = 0L

      for (i <- 0 until param.iteration) {
        var b = 0
        val stackSize = input.size(1) / subModelNumber
        val extraSize = input.size(1) % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val inputBuffer = new Array[Tensor[Float]](parallelism)
        val targetBuffer = new Array[Tensor[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          inputBuffer(b) = input.narrow(1, offset, length)
          targetBuffer(b) = target.narrow(1, offset, length)
          b += 1
        }

        val start = System.nanoTime()
        default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
            () => {
              println(s"running model ${i}")
              val (localModel, localCriterion) = workingModels(i)
              if (inference) {
                localModel.evaluate()
              } else {
                localModel.zeroGradParameters()
                localModel.training()
              }

              val output = localModel.forward(inputBuffer(i))
              if (!inference) {
                localCriterion.forward(output, targetBuffer(i))
                val gradInput = localCriterion.backward(output, targetBuffer(i))
                localModel.backward(inputBuffer(i), gradInput)
              }
            })
        )
        val end = System.nanoTime()

//        // copy multi-model gradient to the buffer
//        default.invokeAndWait(
//          (0 until syncGradParallelNum).map(tid =>
//            () => {
//              val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
//              val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
//              var i = 0
//              while (i < parallelism) {
//                if (i == 0) {
//                  grad.narrow(1, offset + 1, length)
//                    .copy(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
//                } else {
//                  grad.narrow(1, offset + 1, length)
//                    .add(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
//                }
//                i += 1
//              }
//            })
//        )
//        grad.div(parallelism)
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. ")
        if (i >= warmupIteration) {
          timeCost += end - start
          val a = model.getTimes()
          val times = model.getTimesGroupByModuleType()
            .map(v => (v._1, v._2.toDouble, v._2.toDouble))
            .map(v => (v._1, v._2 / 1e9, v._3 / 1e9, (v._2 + v._3)/ 1e9))
          println(times.mkString("\n"))
        } else {
          model.resetTimes()
        }
        println(i)
      }
      logger.info(s"Run ${param.iteration - warmupIteration}, time cost ${timeCost / 1e9}s," +
        s"Average throughput is ${param.batchSize.toDouble * (param.iteration - warmupIteration) / timeCost * 1e9} record / second. ")
    }

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          override def hasNext: Boolean = true

          override def next(): MiniBatch[Float] = {
            MiniBatch(input, labels)
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }
    val p = model.parameters()._1
    val b = p.map(_.nElement()).filter(_ > 100000)
    println(p.map(_.nElement()).mkString("\n"))


    if (!param.inference && param.optim) {
      val optimizer = Optimizer(model, dummyDataSet, null)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
    } else {
      val input = Tensor[Float](param.batchSize, 1, 13, uttLen).rand
      val labels = Tensor[Float](param.batchSize, uttLen / 3).fill(1.0f)
      perfLocal(model, ClassNLLCriterion[Float](), input, labels, param.inference)
    }
    val modules = getModules(model).map{module =>
      val nEle = module.output match {
        case tensor: Tensor[Float] =>
          tensor.nElement()
        case table: Table =>
          table.getState().values.map(_.asInstanceOf[Tensor[Float]].nElement()).sum
      }
      (module.getName(), nEle)
    }
    println(modules.mkString("\n"))
    val itera = param.iteration - warmupIteration
    val times = model.getTimesGroupByModuleType()
      .map(v => (v._1, v._2.toDouble, v._3.toDouble))
      .map(v => (v._1, v._2 / 1e9 / itera, v._3 / 1e9 / itera, (v._2 + v._3)/ 1e9 / itera))
    println(times.mkString("\n"))
    sc.stop()
  }


  def getModules[T: ClassTag](module: Container[_, _, T]): ArrayBuffer[AbstractModule[_, _, T]] = {
    val nodes = ArrayBuffer[AbstractModule[_, _, T]]()
    module.modules.foreach {
      case container: Container[_, _, T] =>
        nodes ++= getModules(container)
      case m =>
        nodes.append(m)
    }

    nodes
  }
}

case class Deepspeech2PerfParam(
  iteration: Int = 1,
  batchSize: Int = 1,
  utt: Int = 3000,
  partition: Int = 1,
  coreNumber: Int = 1,
  module: String = "deepspeech2",
  inputData: String = "random",
  inference: Boolean = true,
  optim: Boolean = false
)
