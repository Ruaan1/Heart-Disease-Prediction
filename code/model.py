import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import BayesianEstimator

df = pd.read_csv('../data/heart_disease_clean.csv')

# shows the structure and variables used
model = BayesianNetwork([
    ('age_cat', 'target'),
    ('chol_cat', 'target'),
    ('bp_cat', 'target'),
    ('thalach_cat', 'target'),
    ('oldpeak_cat', 'target')
])

# fits the model using the BayesianEstimator (BDeu prior is standard)
model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')

print("Fitting is done.")
# prints all CPDs
for cpd in model.get_cpds():
    print(f"\n CPD of {cpd.variable}:")
    print(cpd)