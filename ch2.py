from data import data_dir
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

input_path = data_dir + "01_heights_weights_genders.csv"

heights_weights = read_table(input_path, sep = ',', header = 0)

def my_quantiles(s, prob=(0.0, 0.25, 0.5, 1.0)):

    q = [s.quantile(p) for p in prob]
    print(q)
    return Series(q, index=prob)

plt.figure()
heights = heights_weights["Height"]
bins1 = np.arange(heights.min(), heights.max(), 1.0)
heights.hist(bins=bins1, fc='steelblue')
plt.title('sightings\nAll years in data')
plt.savefig('heights.png')

bins001 = np.arange(heights.min(), heights.max(), 0.001)
heights.hist(bins=bins001, fc="steelblue")
plt.title("bins 0.01")
plt.savefig("height001.png")

#kernel density
density = gaussian_kde(heights.values)
fig = plt.figure()
plt.plot(np.sort(heights.values), density(np.sort(heights.values)))
fig.savefig("kdeheight.png")

heights_m = heights[heights_weights["Gender"] == "Male"].values
heights_f = heights[heights_weights["Gender"] == "Female"].values
density_m = gaussian_kde(heights_m)
density_f = gaussian_kde(heights_f)
plt.plot(np.sort(heights_m), density_m(np.sort(heights_m)), label="Male")
plt.plot(np.sort(heights_f), density_f(np.sort(heights_m)), label="Female")
plt.legend()
plt.savefig("femalemale.png")


fix, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (9, 6))
plt.subplots_adjust(hspace=0.1)
axes[0].plot(np.sort(heights_f), density_f(np.sort(heights_f)), label="Female")
axes[0].xaxis.tick_top()
axes[0].legend()
axes[1].plot(np.sort(heights_m), density_m(np.sort(heights_m)), label="Male")
axes[1].legend()
plt.savefig("inonefig.png")


#
# if __name__ == "__main__":
#     a = heights_weights
#     print(a)


