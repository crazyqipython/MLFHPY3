import pandas as pd
from unittest import mock
import unittest
import numpy as np

birds_df = pd.read_csv('tests/birds_sm.csv', index_col='Species')

def count_birds(birds_df):
    """count the number of birds"""
    years = birds_df.columns
    results_df = pd.DataFrame(index=['Mean'], columns=years)

    for year in years:  # 5
          birds_this_year = birds_df[year]
          sum_counts = birds_this_year.sum()  # 6
          species_seen = (birds_this_year > 0).sum()

          if species_seen == 0:  # 7
              results_df[year] = 0
          else:
              results_df[year] = sum_counts / species_seen
    return results_df

results_df = count_birds(birds_df)
results_df.to_csv('birds_results.csv')

class TestPandas(unittest.TestCase):

    def setUp(self):
        self.birds_df = birds_df

    def test_count_birds(self):
         input_df = pd.DataFrame([[0,2],[0,4]],
            index=['Sp1', 'Sp2'], columns=['2010', '2011'])
         result = count_birds(input_df)
         np.testing.assert_array_equal(result['2010'], 0)

if __name__ == "__main__":
    unittest.main()