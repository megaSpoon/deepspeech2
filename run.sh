#!/usr/bin/env bash
$SPARK_HOME/bin/spark-submit --master local[1] \
   --conf spark.driver.memory=15g \
   --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
   --driver-class-path deepspeech2-bigdl-0.7-SNAPSHOT-jar-with-dependencies.jar \
   --class com.intel.analytics.zoo.pipeline.deepspeech2.example.NervanaInferenceExample \
   target/deepspeech2-bigdl-0.7-SNAPSHOT-jar-with-dependencies.jar  \
   -m weights/ \
   -d data/1462-170145-0004.flac -n 1 -p 1 -s 30
