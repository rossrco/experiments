from operator import add, mul, sub, truediv
from sklearn.model_selection import train_test_split
from numpy import isnan, isinf, isneginf

def update_n_best(best_columns, n_best, current_score, transformation):
    worst_score = 0
    best_columns[current_score] = transformation

    if len(best_columns) > n_best:
        worst_score = min(best_columns.keys())
        best_columns.pop(worst_score)
    return best_columns

def compose_extremum_feature():
    pass

def compose_feature(operator, column1, column2):
    feature = {}
    feature['transformation_function'] = lambda x1, x2: operator(x1, x2)
    feature['column_name'] = '%s_%s_%s' % (column1, operator.__name__, column2)
    feature['x1'] = column1
    feature['x2'] = column2
    return feature

def apply_feature(df, feature, replace_inf):
    res = feature['transformation_function'](df[feature['x1']], df[feature['x2']]).reshape(-1, 1)
    res[isinf(res)] = replace_inf
    res[isneginf(res)] = replace_inf
    return res

def derive_feature_combinations(df, y, operators, best_columns, model, scorer, n_best, replace_inf):
    columns = df.columns

    for c1 in columns:
        for c2 in columns:
            if c1 != c2:
                for o in operators:
                    feature = compose_feature(o, c1, c2)
                    res = apply_feature(df, feature, replace_inf)
                    res_train, res_test, y_train, y_test = train_test_split(res, y, test_size = 0.3, random_state = 42)
                    model.fit(res_train, y_train)
                    y_pred = model.predict(res_test)
                    #y_pred_proba = self.model.predict_proba(res_test)
                    current_score = scorer(y_test, y_pred, average = 'weighted')
                    best_columns = update_n_best(best_columns, n_best, current_score, feature)
    return best_columns

class FeatureEngineer:
    def __init__(self, model, scorer, functions = [add, mul, sub, truediv, max, min], n_best = 5, replace_inf = 0):
        self.extrema = [i for i in functions if i in (max, min)]
        if self.extrema:
            for o, i in enumerate(self.extrema):
                functions.remove(i)
        self.operators = functions
        self.model = model
        self.scorer = scorer
        self.n_best = min(n_best, 1000)
        self.replace_inf = replace_inf
        self.best_columns = {}

    def fit(self, df, y):
        columns = df.columns

        #if self.extrema:
        #    for e in self.extrema:
        #        for c1 in columns:
        #            for c2 in columns:
        #                for o in self.operators:
        #                    transformation_function = lambda x1, x2: o(e(x1), x2)
        #                    extremum = e(df[c1])
        #                    res = o(extremum, df[c2]).reshape(-1, 1)
        #                    column_name = '%s_(%s)_%s_%s' % (e.__name__, c1, o.__name__, c2)
        #                    transformation = {
        #                    'transformation_function' : transformation_function,
        #                    'x1' : c1,
        #                    'x2' : c2,
        #                    'column_name' : column_name
        #                    }
        #                    res_train, res_test, y_train, y_test = train_test_split(res, y, test_size = 0.3, random_state = 42)
        #                    self.model.fit(res_train, y_train)
        #                    y_pred = self.model.predict(res_test)
        #                    current_score = self.scorer(y_test, y_pred, average = 'weighted')
        #                    self.best_columns = update_n_best(self.best_columns, self.n_best, current_score, transformation)

        self.best_columns = derive_feature_combinations(df, y, self.operators,
        self.best_columns, self.model, self.scorer, self.n_best,
        self.replace_inf)

    def transform(self, df):
        for k, trans in self.best_columns.items():
            df[trans['column_name']] = trans['transformation_function'](df[trans['x1']], df[trans['x2']])
