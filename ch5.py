import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.formula.api import ols

from data import data_dir

heights_weights_path = open(os.path.join(data_dir) + "01_heights_weights_genders.csv")
sites_path = open(os.path.join(data_dir) + "top_1000_sites.tsv")
top_1k_sites = pd.read_csv(sites_path, sep = '\t')
heights_weights = pd.read_csv(heights_weights_path)

fit_reg = ols(formula='Weight ~Height', data = heights_weights).fit()
print(fit_reg.summary())

fit_reg_log = ols(formula='np.log(Weight) ~np.log(Height)', data = heights_weights).fit()
print(fit_reg_log.summary())

pred_weights = fit_reg.predict()

heights = heights_weights['Height']
weights = heights_weights['Weight']
plt.figure(figsize=(8,8))
plt.plot(heights, weights, '.', mfc = 'white', alpha = .2)
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.plot(heights, pred_weights, '-k')
# plt.savefig('wh.png')

print(top_1k_sites.head())
kde_model = sm.nonparametric.KDEUnivariate(np.log(top_1k_sites['PageViews'].values))
kde_model.fit()
plt.plot(kde_model.support, kde_model.density)
plt.xlabel('Page Views (log)')
plt.savefig('pageview.png')

# Univariate model
pageview_fit = ols('np.log(PageViews) ~np.log(UniqueVisitors)', data=top_1k_sites).fit()
print(pageview_fit.summary())

plt.figure(figsize=(8,8))
plt.plot(np.log(top_1k_sites['UniqueVisitors']), np.log(top_1k_sites['PageViews']),
         '.', mfc = 'white')
plt.plot(np.log(top_1k_sites['UniqueVisitors']), pageview_fit.predict(), '-k')
plt.xlabel('Unique Visitor(log)')
plt.ylabel('Page Views(log)')
plt.savefig("uniquevisitor.png")

# Multivariate model
model = 'np.log(PageViews) ~ np.log(UniqueVisitors) + HasAdvertising  + InEnglish'
pageview_fit_multi = ols(model, top_1k_sites).fit()
print(pageview_fit_multi.summary())
