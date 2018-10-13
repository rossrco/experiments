import pandas as pd
from collections import deque
from operator import add, mul, sub, truediv

class FeatureEngineer:
    def __init__(self, splitter, model, scorer, functions = [add, mul, sub, truediv, max, min], n_best = 5):
        self.extrema = [i for i in functions if i in (max, min)]
        if self.extrema:
            for o, i in enumerate(self.extrema):
                functions.remove(i)
        self.operators = functions
        self.splitter = splitter
        self.model = model
        self.scorer = scorer
        self.n_best = n_best
        self.best_cols = {}

    def fit_transform(self, df, y):
        columns = df.columns
        best_score = 0

        if self.extrema:
            for e in self.extrema:
                for c1 in columns:
                    for c2 in columns:
                        for o in self.operators:
                            extremum = e(df[c1])
                            res = o(extremum, df[c2])
                            colname = '%s_(%s)_%s_%s' % (e.__name__, c1, o.__name__, c2)
                            res = res.to_frame(name = colname)
                            res_train, res_test, y_train, y_test = self.splitter(res, y, test_size = 0.3, random_state = 42)
                            self.model.fit(res_train, y_train)
                            y_pred = self.model.predict(res_test)
                            current_score = self.scorer(y_test, y_pred, average = 'weighted')
                            self.best_cols[current_score] = res

        for c1 in columns:
            for c2 in columns:
                if c1 != c2:
                    for o in self.operators:
                        colname = '%s_%s_%s' % (c1, o.__name__, c2)
                        res = pd.DataFrame(o(df[c1], df[c2]), columns = [colname])
                        res_train, res_test, y_train, y_test = self.splitter(res, y, test_size = 0.3, random_state = 42)
                        self.model.fit(res_train, y_train)
                        y_pred = self.model.predict(res_test)
                        #y_pred_proba = self.model.predict_proba(res_test)
                        current_score = self.scorer(y_test, y_pred, average = 'weighted')
                        self.best_cols[current_score] = res

        for i, s in enumerate(sorted(self.best_cols, reverse = True)):
            if i < self.n_best:
                df = df.join(self.best_cols[s])
        return df
