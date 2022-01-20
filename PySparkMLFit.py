import operator
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def vector_assembler() -> VectorAssembler:
    features = ["user_type_index", "platform_index", "duration", "item_info_events", "select_item_events",
                "make_order_events", "events_per_min"]
    return VectorAssembler(inputCols=features, outputCol="features")


def build_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(labelCol=LABEL_COL,
                                             predictionCol="prediction",
                                             metricName="f1")


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features")


def build_gbt_classifier() -> GBTClassifier:
    return GBTClassifier(labelCol=LABEL_COL, featuresCol="features")


def build_decision_tree_classifier() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(labelCol=LABEL_COL, featuresCol="features")


def model_params() -> []:
    rf = build_random_forest()
    gbt = build_gbt_classifier()
    return ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 6, 7]) \
        .addGrid(gbt.maxBins, [4, 5, 6]) \
        .build()
#.addGrid(rf.maxDepth, [5, 6, 7]) \
        #.addGrid(rf.maxBins, [4, 5, 6]) \
        #.addGrid(rf.numTrees, [20, 40, 80]) \


def pipeline(my_model):
    user_type_index = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    platform_index = StringIndexer(inputCol='platform', outputCol="platform_index")
    feature = vector_assembler()
    return Pipeline(stages=[user_type_index, platform_index, feature, my_model])


def crossval(pipeline, evaluator, model_param):
    return CrossValidator(estimator=pipeline,
                          estimatorParamMaps=model_param,
                          evaluator=evaluator,
                          numFolds=10)


def process(spark, data_path='6_1/session-stat.parquet', model_path='cv_model'):
    """
    Основной процесс задачи. Обучает модель.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    train_df = spark.read.parquet(data_path)
    my_model = build_random_forest()
    pipe = pipeline(my_model)
    evaluator = build_evaluator()
    model_param = model_params()
    cv = crossval(pipe, evaluator, model_param)
    cv_model = cv.fit(train_df)
    cv_model.bestModel.write().overwrite().save(model_path)


def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
