################################################################################
# Havent done a ton of work with SKLearn lately and would like to more         #
# effectively analyze data. Might split up the preprocessor and final pipeline #
# generation in the model generation class but its a good start                #
# TODO: need to get some comments in and some test cases created.              #
################################################################################

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



class BaseProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
    return None
    
  def fit(self, X=None, y=None):
    return self

  def transform(self, X=None):
    return self

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self._attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self._attribute_names].values


class MeanVarScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._means = None
        self._stds = None

    def fit(self, X=None, y=None):
        try:
            X = X.to_numpy()
            self._means = X.mean(axis=0, keepdims=True)
            self._stds = X.std(axis=0, keepdims=True)
            return self

        except Exception as err:
            print(f'Exception occurred fitting data: {err}')


    def transform(self, X=None, y=None):
        try:
            X[:] = (X.to_numpy() - self._means) / self._stds
            return X

        except Exception as err:
            print(f'Error occured transforming data: {err}')


class CustomModelTrain:

    def __init__(self, df: pd.DataFrame,
                X: list,
                y: str,
                num_pipe : sklearn.pipeline.Pipeline,
                cat_pipe : sklearn.pipeline.Pipeline,
                explicit_num_cols=[],
                explicit_cat_cols=[],
                model=None,
                test_set=0.1):

        self._df = df
        self._params = x
        self._target = y
        self._num_pipe = num_pipe
        self._cat_pipe = cat_pipe
        self._explicit_num_cols = explicit_num_cols
        self._explicit_cat_cols = explicit_cat_cols
        self._model = model
        self._test_set = test_set
        self._preprocess_pipe = self.generate_feature_union()
        self._final_model = self.full_pipeline()




    def get_df(self):
        return self._df

    def get_numeric_pipeline(self):
        return self._num_pipe

    def get_categorical_pipeline(self):
        return self._cat_pipe

    def get_full_pipeline(self):
        return self._full_pipe

    def get_processed_df(self):
        pass

    def get_model(self):
        return self._model

    def get_test_set_size(self):
        return self._test_set

    def get_model(self):
        return self._model



    #column selection functions
    def select_numeric_cols(self):
        return [col for col in self._df if (df[col].dtype == 'int64' or \
        df[col].dtype == 'float64')]

    def select_categorical_cols(self):
        return [col for col in self._df if (df[col].dtype == 'bool' or \
        df[col].dtype == 'category')]


    def generate_feature_union(self):
        num_cols = self._explicit_num_cols if len(self._explicit_num_cols) > 0 \
        else self.select_numeric_cols()

        cat_cols = self._explicit_cat_cols if len(self._explicit_cat_cols) > 0 \
        else self.select_categorical_cols()

        self._num_pipe.steps.insert(0,['selector',DataFrameSelector(num_cols)])
        self._cat_pipe.steps.insert(0,['selector',DataFrameSelector(cat_cols)])

        return FeatureUnion(transformer_list = [
        ('num_pipeline',self._num_pipe),
        ('cat_pipe', self._cat_pipe)
        ])


    def full_pipeline(self):
        X = self._df[self._params]
        y = self._df[self._target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size = self._test_set)

        full_pipeline = Pipeline(steps=[
        ('preprocess_pipeline', self._preprocess_pipe   ),
        ('model', self._model)])

        model = full_pipeline.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return model
