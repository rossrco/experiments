import pandas as pd
from collections import deque
from operator import add, mul, sub, truediv

class FeatureEngineer:
    def __init__(self, splitter, model, scorer, operators = {'add' : add, 'mul' : mul, 'sub' : sub, 'div' : truediv}):
        self.operators = operators
        self.splitter = splitter
        self.model = model
        self.scorer = scorer

    def fit_transform(self, df, y):
        columns = df.columns
        best_score = 0
        best_cols = []
        for c1 in columns:
            for c2 in columns:
                if c1 != c2:
                    for o_n, o in self.operators.items():
                        colname = '%s_%s_%s' % (c1, o_n, c2)
                        res = pd.DataFrame(o(df[c1], df[c2]), columns = [colname])
                        res_train, res_test, y_train, y_test = self.splitter(res, y, test_size = 0.3, random_state = 42)
                        self.model.fit(res_train, y_train)
                        y_pred = self.model.predict(res_test)
                        y_pred_proba = self.model.predict_proba(res_test)
                        current_score = self.scorer(y_test, y_pred, average = 'weighted')
                        if current_score > best_score:
                            best_score = current_score
                            best_cols.append(res)
        for c in best_cols:
            df = df.join(c)
        return df
