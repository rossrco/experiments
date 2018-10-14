from operator import add, mul, sub, truediv

def update_n_best(best_cols, n_best, current_score, res_df):
    worst_score = 0
    best_cols[current_score] = res_df

    if len(best_cols) > n_best:
        worst_score = min(best_cols.keys())
        best_cols.pop(worst_score)
    return best_cols

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
        self.n_best = min(n_best, 1000)
        self.best_cols = {}

    def fit(self, df, y):
        columns = df.columns
        best_score = 0
        worst_score = 0

        if self.extrema:
            for e in self.extrema:
                for c1 in columns:
                    for c2 in columns:
                        for o in self.operators:
                            transformation_f = lambda x1, x2: o(e(x1), x2)
                            extremum = e(df[c1])
                            res = o(extremum, df[c2]).reshape(-1, 1)
                            colname = '%s_(%s)_%s_%s' % (e.__name__, c1, o.__name__, c2)
                            transformation = {
                            'transformation_function' : transformation_f,
                            'x1' : c1,
                            'x2' : c2,
                            'column_name' : colname
                            }
                            res_train, res_test, y_train, y_test = self.splitter(res, y, test_size = 0.3, random_state = 42)
                            self.model.fit(res_train, y_train)
                            y_pred = self.model.predict(res_test)
                            current_score = self.scorer(y_test, y_pred, average = 'weighted')
                            self.best_cols = update_n_best(self.best_cols, self.n_best, current_score, transformation)

        for c1 in columns:
            for c2 in columns:
                if c1 != c2:
                    for o in self.operators:
                        transformation_f = lambda x1, x2: o(x1, x2)
                        colname = '%s_%s_%s' % (c1, o.__name__, c2)
                        transformation = {
                        'transformation_function' : transformation_f,
                        'x1' : c1,
                        'x2' : c2,
                        'column_name' : colname
                        }
                        res = o(df[c1], df[c2]).reshape(-1, 1)
                        res_train, res_test, y_train, y_test = self.splitter(res, y, test_size = 0.3, random_state = 42)
                        self.model.fit(res_train, y_train)
                        y_pred = self.model.predict(res_test)
                        #y_pred_proba = self.model.predict_proba(res_test)
                        current_score = self.scorer(y_test, y_pred, average = 'weighted')
                        self.best_cols = update_n_best(self.best_cols, self.n_best, current_score, transformation)

    def transform(self, df):
        for k, trans in self.best_cols.items():
            df[trans['column_name']] = trans['transformation_function'](df[trans['x1']], df[trans['x2']])
