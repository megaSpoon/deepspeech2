Steps to run

### Download ds2 model :

nervna model:
https://drive.google.com/drive/folders/0B9zID9CU9HQeM1M1SXpmN3oyLUU

download it and unzip into the ***data*** folder

### build jar
In repo main directory, run ```mvn clean package``` and get deepspeech2-bigdl-0.6-SNAPSHOT-jar-with-dependencies.jar

### Run Nervanainference with example:

set SPARK_HOME(version recommended: 2.0.1) and put deepspeech2-bigdl-0.6-SNAPSHOT-jar-with-dependencies.jar in current repo 
main directory

run the following script for nervana ds2 inference with bigdl:

```shell
 $SPARK_HOME/bin/spark-submit --master local[1] \
   --conf spark.driver.memory=15g \
   --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
   --driver-class-path deepspeech2-bigdl-0.6-SNAPSHOT-jar-with-dependencies.jar \
   --class com.intel.analytics.zoo.pipeline.deepspeech2.example.NervanaInferenceExample \
   deepspeech2-bigdl-0.6-SNAPSHOT-jar-with-dependencies.jar  \
   -m data/ \
   -d data/1462-170145-0004.flac -n 1 -p 1 -s 30

   ```



