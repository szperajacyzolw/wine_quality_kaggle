'''Tekne Consulting blogpost --- teknecons.com'''


from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import os
import joblib
import ppscore as pps
from sklearn import preprocessing as pre
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve,
                             balanced_accuracy_score, recall_score, precision_score)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint, zscore, median_abs_deviation
import scikitplot as skplt


'''initial tests'''
# treshold = 7, no dupliacates removal, no outliears removal, no stratify in train_test_split
# default RandomForestClassifier: Score: 0.9145833333333333, F1 score: 0.5454545454545455!!!
# that means, the most popular approach with '.score()' evaluation is not relevant(inbalanced data)
# treshold = 6, no dupliacates removal, no outliears removal, no stratify in train_test_split
# default RandomForestClassifier: F1 score:0.8226415094339623


'''data source initialization'''


this_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(this_dir, 'winequality-red.csv')
quality_data = pd.read_csv(data_path)
# quality_data.drop_duplicates(inplace=True)
# duplicates seems to be relevant and may exist due to precess satbility, so I'm leaving them


'''predictive score matrix'''


matrix_pp = pps.matrix(quality_data)[['x', 'y', 'ppscore']].pivot(
    columns='x', index='y', values='ppscore')
fig0, ax0 = plt.subplots()
plot0 = sns.heatmap(matrix_pp, annot=True, ax=ax0)
# plt.show()
# most of features have no predictive power, so let's drop them :)


'''dependent and independet variables separation and cleaning'''


def robust_zscore(d: 'DataFrame'):
    a = np.asanyarray(d)
    median = np.median(a, axis=0)
    mad = median_abs_deviation(a, axis=0)
    z = (a - median) / mad
    return z
# robust z-score is distribution agnostic but is far more sensitive
# 'normal' z-score works fine only for normally distributed variable


depend = pd.DataFrame([0 if q < 6 else 1 for q in quality_data['quality']],
                      columns=['quality_class'])
independ = quality_data.drop('quality', axis=1)
print(len(depend), len(depend[(np.abs(zscore(independ) < 3).all(axis=1))]),
      len(depend[(np.abs(robust_zscore(independ) < 8).all(axis=1))]))  # compare scoring sensitivity

depend = depend[(np.abs(robust_zscore(independ) < 8).all(axis=1))]  # outliers out by robust_zscore
independ = independ[(np.abs(robust_zscore(independ) < 8).all(axis=1))
                    ]  # outliers out by robust_zscore
independ = independ.loc[:, ['alcohol', 'density', 'sulphates',
                            'total sulfur dioxide', 'volatile acidity']]  # leave only important columns
# after performed cleaning, there is no duplicates in the data! Even with no .drop_duplicates()

'''some data exploration'''


profile = ProfileReport(independ.join(depend), title='red wine quality', explorative=True)
profile.to_file(os.path.join(this_dir, 'profie_report_wine_clean.html'))


'''predictive score matrix after data cleaning'''


matrix_pp = pps.matrix(depend.join(independ))[['x', 'y', 'ppscore']].pivot(
    columns='x', index='y', values='ppscore')
fig1, ax1 = plt.subplots()
plot1 = sns.heatmap(matrix_pp, annot=True, ax=ax1)
# plt.show()
# much prettier :)

'''scaling of variables'''


scaler = pre.StandardScaler()
scaled = scaler.fit_transform(independ)
independ_sc = pd.DataFrame(scaled, columns=independ.columns)


'''splitting data for training and test'''


itrain, itest, dtrain, dtest = train_test_split(
    independ_sc, depend, stratify=depend, random_state=12, test_size=0.3)
# stratify makes scoring more relevant


'''gradien boosted trees'''


def gb_params():
    loss = ['deviance', 'exponential']
    learning_rate = uniform(0.0001, 0.2)
    n_estimators = randint(10, 210)
    subsample = uniform(0.2, 0.8)
    criterion = ['friedman_mse', 'mse']
    min_samples_split = randint(2, 20)
    min_samples_leaf = randint(2, 20)
    max_depth = randint(2, 10)

    grid = {'loss': loss,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'criterion': criterion}
    return(grid)


'''random forest'''


def rf_params():
    n_estimators = randint(10, 210)
    max_features = ['auto', 'sqrt']
    max_depth = [2, 5, 10, None]
    min_samples_split = randint(2, 20)
    min_samples_leaf = randint(2, 20)
    bootstrap = [True, False]
    criterion = ['entropy', 'gini']

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'criterion': criterion}
    return(grid)


'''multi layer perception'''


def mlpc_params():
    hidden_layer_sizes = [(i, j) for i in range(4, 11) for j in range(2, 5)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['sgd', 'adam']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    max_iter = randint(1000, 10000)

    grid = {'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'learning_rate': learning_rate,
            'max_iter': max_iter}
    return(grid)


'''nearest neighbours'''


def knc_params():
    n_neighbors = randint(1, 11)
    weights = ['uniform', 'distance']
    algorithm = ['ball_tree', 'kd_tree']
    leaf_size = randint(5, 50)
    p = uniform(1, 4)

    grid = {'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p}
    return(grid)


''' logistic regression'''


def logit_params():
    penalty = ['l1', 'l2']
    class_weight = [None, 'balanced']
    solver = ['liblinear', 'saga']
    max_iter = randint(50, 500)

    grid = {'penalty': penalty,
            'class_weight': class_weight,
            'solver': solver,
            'max_iter': max_iter}
    return(grid)


'''searching for optimal hyperparameters'''


def optimizer(clf: 'classifier object', grid: 'parmas dict'):
    clf = clf()
    opt_clf = RandomizedSearchCV(estimator=clf, param_distributions=grid,
                                 n_iter=100, cv=10, verbose=0, random_state=123,
                                 n_jobs=-1, scoring='roc_auc')

    opt_clf.fit(itrain, np.ravel(dtrain))
    return(opt_clf)


'''final scoring'''


def final_score(clf: 'classifier with optimized params'):
    predicted = clf.predict(itest)
    p_score = precision_score(dtest, predicted)
    r_score = recall_score(dtest, predicted)
    f1 = f1_score(dtest, predicted)
    rauc_score = roc_auc_score(dtest, clf.predict_proba(itest)[:, 1])
    precision, recall, _ = precision_recall_curve(dtest, clf.predict_proba(itest)[:, 1])
    pr_auc_score = auc(recall, precision)
    balanced_accuracy = balanced_accuracy_score(dtest, predicted)
    scores_dict = {'precision_score': p_score, 'recall_score': r_score, 'f1_score': f1,
                   'roc_auc_score': rauc_score, 'precision_recal_auc': pr_auc_score,
                   'balanced_accuracy_score': balanced_accuracy}
    return(scores_dict)


gb = optimizer(GradientBoostingClassifier, gb_params())
gbs = final_score(gb)
rf = optimizer(RandomForestClassifier, rf_params())
rfs = final_score(rf)
mlpc = optimizer(MLPClassifier, mlpc_params())
mlpcs = final_score(mlpc)
knc = optimizer(KNeighborsClassifier, knc_params())
kncs = final_score(knc)
# logistic regression as the simplest benchmark
logit = optimizer(LogisticRegression, logit_params())
logits = final_score(logit)
# save all the models
joblib.dump(gb, os.path.join(this_dir, 'gb.sav'))
joblib.dump(rf, os.path.join(this_dir, 'rf.sav'))
joblib.dump(mlpc, os.path.join(this_dir, 'mlpc.sav'))
joblib.dump(knc, os.path.join(this_dir, 'knc.sav'))
joblib.dump(logit, os.path.join(this_dir, 'logit.sav'))
# dict that stores optimizer params and scoring
optimizers_data = {'gb': (gb.best_estimator_, gb.scorer_, gbs), 'rf': (rf.best_estimator_, rf.scorer_, rfs),
                   'mlpc': (mlpc.best_estimator_, mlpc.scorer_, mlpcs), 'knc': (knc.best_estimator_, knc.scorer_, kncs),
                   'logit': (logit.best_estimator_, logit.scorer_, logits)}
joblib.dump(optimizers_data, os.path.join(this_dir, 'optimizers_data.sav'))
# ditct with only scores for further processing
scores = {key: value[-1] for key, value in optimizers_data.items()}
df_scores = pd.DataFrame.from_dict(scores)


'''results summary on graphs'''


dtest_flat = np.ravel(dtest)
fig2, ax2 = plt.subplots()
plot2 = sns.heatmap(data=df_scores, annot=True, ax=ax2)
fig3, ((ax31, ax32), (ax33, ax34), (ax35, ax36)) = plt.subplots(
    nrows=3, ncols=2, constrained_layout=True)
skplt.metrics.plot_roc(dtest_flat, gb.predict_proba(
    itest), title='GradientBoostingClassifier ROC', ax=ax31)
skplt.metrics.plot_roc(dtest_flat, rf.predict_proba(
    itest), title='RandomForestClassifier ROC', ax=ax32)
skplt.metrics.plot_roc(dtest_flat, mlpc.predict_proba(itest), title='MLPClassifier ROC', ax=ax33)
skplt.metrics.plot_roc(dtest_flat, knc.predict_proba(
    itest), title='KNeighborsClassifier ROC', ax=ax34)
skplt.metrics.plot_roc(dtest_flat, logit.predict_proba(itest),
                       title='LogisticRegression ROC', ax=ax35)
fig4, ((ax41, ax42), (ax43, ax44), (ax45, ax46)) = plt.subplots(
    nrows=3, ncols=2, constrained_layout=True)
skplt.metrics.plot_ks_statistic(dtest_flat, gb.predict_proba(
    itest), title='GradientBoostingClassifier KS', ax=ax41)
skplt.metrics.plot_ks_statistic(dtest_flat, rf.predict_proba(
    itest), title='RandomForestClassifier KS', ax=ax42)
skplt.metrics.plot_ks_statistic(dtest_flat, mlpc.predict_proba(itest),
                                title='MLPClassifier KS', ax=ax43)
skplt.metrics.plot_ks_statistic(dtest_flat, knc.predict_proba(itest),
                                title='KNeighborsClassifier KS', ax=ax44)
skplt.metrics.plot_ks_statistic(dtest_flat, logit.predict_proba(itest),
                                title='LogisticRegression KS', ax=ax45)

ax36.set_visible(False)
ax46.set_visible(False)
plt.show()


'''model axplanation'''


explainer = shap.KernelExplainer(knc.predict, shap.sample(itest, 200))
shap_values = explainer.shap_values(itest.iloc[:200, :])
shap.summary_plot(shap_values, itest.iloc[:200, :], plot_type='layered_violin')


print(f'\
optimized GradientBoostingClassifier. Scoring: \n{gbs}\nParams:\n{gb.best_estimator_}\n\
optimized RandomForestClassifier. Scoring: \n{rfs}\nParams: \n {rf.best_estimator_}\n\
optimized MLPClassifier. Scoring: \n{mlpcs} \nParams:\n{mlpc.best_estimator_}\n\
optimized KNeighborsClassifier. Scoring: \n{kncs}\nParams: \n {knc.best_estimator_}\n\
optimized LogisticRegression. Scoring: \n{logits}\nParams:\n{logit.best_estimator_}')
