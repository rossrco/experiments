from operator import add, mul, sub, truediv
from numpy import isnan, isinf, isneginf
from collections import defaultdict

def update_n_best(best_columns, n_best, current_score, transformation):
    worst_score = 0
    best_columns[current_score] = transformation

    if len(best_columns) > n_best:
        worst_score = min(best_columns.keys())
        best_columns.pop(worst_score)
    return best_columns

def apply_feature(df, feature, replace_inf):
    res = feature['transformation_function'](df[feature['x1']], df[feature['x2']]).values.reshape(-1, 1)
    res[isinf(res)] = replace_inf
    res[isneginf(res)] = replace_inf
    res[isnan(res)] = replace_inf
    return res

def get_current_score(x, y, model, scorer, splitter, predict_proba,
scorer_kwargs = None, splitter_kwargs = None):
    if splitter_kwargs:
        x_train, x_test, y_train, y_test = splitter(x, y, **splitter_kwargs)
    else:
        x_train, x_test, y_train, y_test = splitter(x, y)

    model.fit(x_train, y_train)
    if predict_proba:
        y_pred = model.predict_proba(x_test)[:, 1]
    else:
        y_pred = model.predict(x_test)

    if scorer_kwargs:
        current_score = scorer(y_test, y_pred, **scorer_kwargs)
    else:
        current_score = scorer(y_test, y_pred)
    return current_score

def define_operator_generator(aggregates = None):
    if aggregates:
        def feature_item_generator(columns, operators, aggregates):
            for a in aggregates:
                for c1 in columns:
                    for c2 in columns:
                            for o in operators:
                                yield {'a' : a, 'c1' : c1, 'c2' : c2, 'o' : o}
    else:
        def feature_item_generator(columns, operators, aggregates):
            for c1 in columns:
                for c2 in columns:
                    if c1 != c2:
                        for o in operators:
                            yield {'a' : None, 'c1' : c1, 'c2' : c2, 'o' : o}

    return feature_item_generator

def define_feature_composer(aggregates = None):
    if aggregates:
        def feature_composer(feature_properties):
            column1 = feature_properties['c1']
            column2 = feature_properties['c2']
            operator = feature_properties['o']
            aggregate = feature_properties['a']
            feature = {}
            feature['transformation_function'] = lambda x1, x2: operator(aggregate(x1), x2)
            feature['column_name'] = '%s_(%s)_%s_%s' % (aggregate.__name__, column1, operator.__name__, column2)
            feature['x1'] = column1
            feature['x2'] = column2
            return feature
    else:
        def feature_composer(feature_properties):
            column1 = feature_properties['c1']
            column2 = feature_properties['c2']
            operator = feature_properties['o']
            feature = {}
            feature['transformation_function'] = lambda x1, x2: operator(x1, x2)
            feature['column_name'] = '%s_%s_%s' % (column1, operator.__name__, column2)
            feature['x1'] = column1
            feature['x2'] = column2
            return feature
    return feature_composer

def derive_feature_combinations(df, y, operators, best_columns, model, scorer,
splitter, n_best, replace_inf, predict_proba, aggregates = None,
scorer_kwargs = None, splitter_kwargs = None):
    generate_items = define_operator_generator(aggregates)
    compose_feature = define_feature_composer(aggregates)
    columns = df.columns

    for property in generate_items(columns, operators, aggregates):
        feature = compose_feature(property)
        res = apply_feature(df, feature, replace_inf)
        current_score = get_current_score(res, y, model, scorer, splitter, predict_proba, scorer_kwargs, splitter_kwargs)
        best_columns = update_n_best(best_columns, n_best, current_score, feature)

    return best_columns

class FeatureEngineer:
    def __init__(self, model, scorer, splitter,
    functions = {'operators' : [add, mul, sub, truediv],
    'aggregates' : [max, min]}, n_best = 5, replace_inf = 0,
    predict_proba  = False, scorer_kwargs = None, splitter_kwargs = None):
        self.aggregates = functions['aggregates']
        self.operators = functions['operators']
        self.model = model
        self.scorer = scorer
        self.splitter = splitter
        self.n_best = min(n_best, 1000)
        self.replace_inf = replace_inf
        self.predict_proba = predict_proba
        self.scorer_kwargs = scorer_kwargs
        self.splitter_kwargs = splitter_kwargs
        self.aggregates_values = {}
        self.best_columns = {}

    def fit(self, df, y):

        if self.aggregates:
            self.best_columns = derive_feature_combinations(df, y, self.operators,
            self.best_columns, self.model, self.scorer, self.splitter, self.n_best,
            self.replace_inf, self.predict_proba, self.aggregates,
            self.scorer_kwargs, self.splitter_kwargs)

        self.best_columns = derive_feature_combinations(df, y, self.operators,
        self.best_columns, self.model, self.scorer, self.splitter, self.n_best,
        self.replace_inf, self.predict_proba, None, self.scorer_kwargs,
        self.splitter_kwargs)

    def transform(self, df):
        for k, trans in self.best_columns.items():
            df[trans['column_name']] = trans['transformation_function'](df[trans['x1']], df[trans['x2']])
