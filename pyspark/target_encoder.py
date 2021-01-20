from pyspark import keyword_only
from pyspark.sql import types as T
from pyspark.ml.param.shared import Param, Params, TypeConverters, HasPredictionCol, HasInputCol
from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class HasMean(Params):
    mean = Param(Params._dummy(), 'mean', 'mean')

    def __init__(self):
        super(HasMean, self).__init__()

    def setMean(self, value):
        return self._set(mean=value)

    def getMean(self):
        return self.getOrDefault(self.mean)


class HasTargetCol(Params):
    target_col = Param(Params._dummy(), 'target_col', 'target_col', typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasTargetCol, self).__init__()

    def setTargetCol(self, value):
        return self._set(target_col=value)

    def getTargetCol(self):
        return self.getOrDefault(self.target_col)


class TargetEncoder(Estimator, HasInputCol, HasPredictionCol, HasMean, HasTargetCol, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol=None, predictionCol=None, target_col='label'):
        super(TargetEncoder, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    @keyword_only
    def setParams(self, inputCol=None, predictionCol=None, target_col='label'):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        col = self.getInputCol()
        target = self.getTargetCol()
        encoded_name = self.getPredictionCol()

        m = dataset.groupby(col).agg(F.mean(target).cast(T.DoubleType()).alias(encoded_name))

        return TargetEncoderModel(inputCol=col, mean=m, target_col=self.getTargetCol(), predictionCol=self.getPredictionCol())


class TargetEncoderModel(Model, HasInputCol, HasPredictionCol, HasMean, HasTargetCol, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol=None, predictionCol=None, mean=None, target_col=None):
        super(TargetEncoderModel, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs) 

    @keyword_only
    def setParams(self, inputCol=None, predictionCol=None, mean=None, target_col=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)          

    def _transform(self, dataset):
        col = self.getInputCol()
        m = self.getMean()

        return dataset.join(other=m, on=[col], how='left')
