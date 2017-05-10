
import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from sklearn import linear_model
from activelearn.exampleselector.smallest_margin_selector import SmallestMarginSelector

class SmallestMarginSelectorTests(unittest.TestCase):
    def setUp(self):
        labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/seed.csv"), sep='\t')
        self.unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/sample_fvs.csv"), sep='\t')
        self.model = linear_model.LogisticRegression()
        feature_attrs = list(self.unlabeled_dataset.columns)
        feature_attrs.remove('_id')
        feature_attrs.remove('l_ID')
        feature_attrs.remove('r_ID')
        self.model.fit(self.unlabeled_dataset[feature_attrs].values[:3],
                                            labeled_dataset_seed['label'].values[:3])
        
    #testing non batch mode
    def test_select_example(self):
        es = SmallestMarginSelector()
        
        instance_to_be_labeled = es.select_examples(
                                     self.unlabeled_dataset.head(5), self.model, 
                                     ['_id', 'l_ID', 'r_ID'])
  
        assert_equal(1,instance_to_be_labeled.iloc[0]["_id"])
    
    #testing batch mode
    def test_select_examples(self):
        es = SmallestMarginSelector()
        
        instances_to_be_labeled = es.select_examples(
                                      self.unlabeled_dataset.head(5), self.model,
                                      ['_id', 'l_ID', 'r_ID'], 2)

        assert_equal(1,instances_to_be_labeled.iloc[0]["_id"])
        assert_equal(4,instances_to_be_labeled.iloc[1]["_id"])
        
    @raises(TypeError)
    def test_entropy_selector_invalid_unlabeled_dataset(self):
        es = SmallestMarginSelector()
        es.select_examples([], self.model,
                                      ['_id', 'l_ID', 'r_ID'], 2)
    @raises(AssertionError)
    def test_entropy_selector_invalid_exclude_attr(self):
        es = SmallestMarginSelector()
        es.select_examples(self.unlabeled_dataset.head(5), self.model,
                                      ['_id', 'l_ID', 'A_ID'], 2)
if __name__ == '__main__':
    unittest.main()
