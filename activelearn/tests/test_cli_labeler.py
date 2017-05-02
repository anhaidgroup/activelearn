
import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from activelearn.labeler.cli_labeler import CliLabeler


class CliLabelerTests(unittest.TestCase):
    def setUp(self):
        self.table_A = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/table_A.csv"), sep=',')
        self.table_B = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/table_B.csv"), sep=',')
        

    def sample_get_instruction_fn(self):
        banner_str = "Select whether the given below pair is a Match(y) or Non Match(n)" + "\n"
        return banner_str

    def sample_get_example_display_fn(self, examples_to_label, context):
        
        fvs_A_id_attr = 'l_ID'
        fvs_B_id_attr = 'r_ID'
        A_id_attr = 'A.ID'
        B_id_attr = 'B.ID'
        
        A_out_attrs = ['A.ID', 'birth_year','hourly_wage']
        B_out_attrs = ['B.ID', 'birth_year','hourly_wage']
        
        #obtaining the raw representation
        table_A_id = examples_to_label[fvs_A_id_attr]
        table_B_id = examples_to_label[fvs_B_id_attr]
        
        raw_tuple_table_A = context.table_A.where(context.table_A[A_id_attr] == table_A_id).dropna().head(1)
        raw_tuple_table_B = context.table_B.where(context.table_B[B_id_attr] == table_B_id).dropna().head(1)
        
        #'A.ID', 'B.ID', 'l_ID', 'r_ID', ['birth_year','hourly_wage'], ['birth_year','hourly_wage']
        return str(raw_tuple_table_A[A_out_attrs]) + "\n" + str(raw_tuple_table_B[B_out_attrs]) + "\n"
        
    @raises(TypeError)
    def test_cli_labeler_invalid_unlabeled_dataset(self):
        labeler = CliLabeler(self.sample_get_instruction_fn, self.sample_get_example_display_fn)
        context = {"table_A": self.table_A, "table_B":self.table_B}
        label_attr = 'label'
        labeler.label([], context, label_attr)

    @raises(TypeError)
    def test_cli_labeler_invalid_functions(self):
        #passing an object instead of function to CliLabeler
        unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/sample_fvs.csv"), sep='\t')
        labeler = CliLabeler({}, self.sample_get_example_display_fn)
        context = {"table_A": self.table_A, "table_B":self.table_B}
        label_attr = 'label'
        labeler.label(unlabeled_dataset, context, label_attr)
        
#     def test_label(self):
#         eml = CliLabeler(self.default_get_instruction_fn, self.display_tuple_pair_for_label, labels= {"y":0, "n":1}, label_attr='label')
#         eml.label(self.feature_vs)
