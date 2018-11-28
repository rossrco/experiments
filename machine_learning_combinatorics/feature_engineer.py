from operator import add, mul, sub, truediv
from numpy import isnan, isinf, isneginf, inf
from collections import defaultdict

def update_n_best(best_columns, n_best, current_score, feature):
    """
    Takes: best columns dictionary, maximum number of columns to produce,
    current feature's score, current feature.
    Does: adds the current feature to the best columns dictionary. Removes the
    lowest performing feature if the number of items in the dictionary exceeds
    the maximum number of columns to produce.
    Returns: best columns dictionary
    """
    worst_score = 0
    best_columns[current_score] = feature

    if len(best_columns) > n_best:
        worst_score = min(best_columns.keys())
        best_columns.pop(worst_score)
    return best_columns

def apply_feature(df, feature, replace_pos_inf, replace_neg_inf, replace_nan):
    """
    Takes: dataframe, feature dictionary value to replace inf with
    Does: uses the columns from the dataframe required by the feature. Produces
    a new feature based on the transformation function.
    Returns: numpy array representing the values of the new feature.
    """
    res = feature['transformation_function'](df[feature['x1']], df[feature['x2']]).values.reshape(-1, 1)
    res[isinf(res)] = replace_pos_inf
    res[isneginf(res)] = replace_neg_inf
    res[isnan(res)] = replace_nan
    return res

def get_current_score(x, y, model, scorer, splitter, predict_proba,
scorer_kwargs = None, splitter_kwargs = None):
    """
    Takes: feature, target, supervised learning algorithm (e.g. logistic
    regression), testing function (e.g. f1 score), splitting function (e.g.
    train-test split), arguments for the last 3.
    Does: splits the feature and the target into a training and validation sets.
    Fits the model using the feature to the target. Predicts the target's values
    for the validation set. Tests the predictions against the ground truth.
    Returns: the score of the predictions based on the testing function.
    """
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

def define_operator_generator(aggregate_mode):
    """
    Takes: aggregate operators list.
    Does: creates a generator of iterables to be used in a feature. An aggregate
    feature is of the form in prefix notation: operator(aggregate(x1), x2).
    Example: max(x1) / x2. To achieve that, the function iterates over all
    aggregate operators, all columns against all columns and all operators. The
    result is an aggregate operator, column1, column2 and an operator.
    The other form of features is: operator(x1, x2), for example: x1 * x2.
    To produce such features, the function needs to iterate over every column
    against every other column (discarding repeating combinations) and over
    every operator.
    Returns: a generator that yeilds a dictionary of feature components
    (aggregate operator, operator, column1, column2).
    """
    if aggregate_mode:
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

def define_feature_composer(aggregate_mode):
    """
    Takes:
    Does:
    Returns:
    """
    if aggregate_mode:
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
splitter, n_best, replace_pos_inf, replace_neg_inf, replace_nan, predict_proba,
aggregates, aggregate_mode, scorer_kwargs = None, splitter_kwargs = None):
    """
    Takes:
    Does:
    Returns:
    """
    generate_items = define_operator_generator(aggregate_mode)
    compose_feature = define_feature_composer(aggregate_mode)
    columns = df.columns

    for property in generate_items(columns, operators, aggregates):
        feature = compose_feature(property)
        res = apply_feature(df, feature, replace_pos_inf, replace_neg_inf, replace_nan)
        current_score = get_current_score(res, y, model, scorer, splitter, predict_proba, scorer_kwargs, splitter_kwargs)
        best_columns = update_n_best(best_columns, n_best, current_score, feature)

    return best_columns

class Feature:
    def __init__(self, x1, x2, function, name):
        self.x1 = x1
        self.x2 = x2
        self.function = function
        self.__name__ = name

    def apply_feature(self, df, replace_pos_inf, replace_neg_inf, replace_nan):
        res = self.function(df[self.x1], df[self.x2]).values.reshape(-1, 1)
        res[isinf(res)] = replace_pos_inf
        res[isneginf(res)] = replace_neg_inf
        res[isnan(res)] = replace_nan
        return res

class FeatureEngineer:
    def __init__(self, model, scorer, splitter, operators = [add, mul, sub, truediv],
    aggregates = [], aggregate_mode = False, n_best = 5, replace_pos_inf = 10e20,
    replace_neg_inf = -10e20, replace_nan = 0, predict_proba  = False,
    scorer_kwargs = None, splitter_kwargs = None):
        self.aggregates = aggregates
        self.operators = operators
        self.aggregate_mode = aggregate_mode
        self.model = model
        self.scorer = scorer
        self.splitter = splitter
        self.n_best = min(n_best, 1000)
        self.replace_pos_inf = replace_pos_inf
        self.replace_neg_inf = replace_neg_inf
        self.replace_nan = replace_nan
        self.predict_proba = predict_proba
        self.scorer_kwargs = scorer_kwargs
        self.splitter_kwargs = splitter_kwargs
        self.aggregates_values = {}
        self.best_columns = {}

    def fit(self, df, y):
        """
        Takes:
        Does:
        Returns:
        """
        self.best_columns = derive_feature_combinations(df, y, self.operators,
        self.best_columns, self.model, self.scorer, self.splitter, self.n_best,
        self.replace_pos_inf, self.replace_neg_inf, self.replace_nan,
        self.predict_proba, self.aggregates, self.aggregate_mode,
        self.scorer_kwargs, self.splitter_kwargs)

    def transform(self, df):
        """
        Takes:
        Does:
        Returns:
        """
        for k, trans in self.best_columns.items():
            df[trans['column_name']] = trans['transformation_function'](df[trans['x1']], df[trans['x2']])

        for to_replace, value in zip((inf, -inf),
        (self.replace_pos_inf, self.replace_neg_inf)):
            df = df.replace(to_replace, value)

        return df
