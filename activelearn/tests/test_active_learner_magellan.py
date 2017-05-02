# import unittest
#  
# from nose.tools import *
# from mock import MagicMock
#  
# import pandas as pd
# import numpy as np
# import operator
# import os
# import py_entitymatching as em
# from sklearn import linear_model
# from sklearn.ensemble import RandomForestClassifier
# from activelearn.exampleselector.entropy_based_example_selector import EntropyBasedExampleSelector
# from activelearn.labeler.cli_labeler import CliLabeler
# from activelearn.activelearner.active_learner import ActiveLearner
# from activelearn.utils.validation import validate_attr
#  
#  
# def match(ltup, rtup):
#     return False
#  
# class ActiveLearnerTests(unittest.TestCase):
#      
#     def sample_get_instruction_fn(self, context):
#         return "Can you enter the label y or n?"
#  
#     def get_example_display_fn(self, example, context):
#         table_a = context["dataset_a"]
#         table_b = context["dataset_b"]
#         example_A = table_a[table_a["id"] == example["ltable_id"]]
#         example_B = table_b[table_b["id"] == example["rtable_id"]]
#         return str(example_A) + "\n" + str(example_B)
#      
#     def get_feats(self, sample_A, sample_B):
#         match_t = em.get_tokenizers_for_matching()
#         match_s = em.get_sim_funs_for_matching()
#         atypes1 = em.get_attr_types(sample_A)
#         atypes2 = em.get_attr_types(sample_B)
#         match_c = em.get_attr_corres(sample_A, sample_B)
#         match_c['corres'] = [('title', 'song'),('artist_name', 'artists')]
#         match_f = em.get_features(sample_A, sample_B, atypes1, atypes2, match_c, match_t, match_s)
#         return match_f
#      
#     def train_fvs(self, dev_set, match_f, i_attrs_after=None):
#         ''' get feature vectors for train set '''
#         H = em.extract_feature_vecs(dev_set, feature_table=match_f, attrs_before = ['_id', 'ltable_id', 'rtable_id'], attrs_after=i_attrs_after)
#         return H
#  
#  
#      
#     def bb_block(self, sample_A, sample_B):
#         bb = em.BlackBoxBlocker()
#         bb.set_black_box_function(match)
#         bbC = bb.block_tables(sample_A, sample_B, l_output_attrs=['id', 'title', 'artist_name', 'year'], r_output_attrs=['id','title', 'year', 'episode','song', 'artists'], show_progress=True)
#         #bbC.to_csv('block_result.csv', index = False, encoding='utf-8')
#         return bbC
#  
#     def setUp(self):
#         sample_A = em.read_csv_metadata('data/songs.csv', key='id')
#         sample_B = em.read_csv_metadata('data/tracks.csv', key='id')
#          
#         #test_set = em.read_csv_metadata(argv[4], key='_id',ltable=sample_A, rtable=sample_B, fk_ltable='ltable_id', fk_rtable='rtable_id')
#      
# #         dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/songs.csv')).head(1000)
# #         dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/tracks.csv')).head(1000)
#          
#         # labeled data,containing labeled raw examples
#         self.labeled_dataset_seed = em.read_csv_metadata(os.path.join(os.path.dirname(__file__), 'data/I.csv'), key='_id',ltable=sample_A, rtable=sample_B, fk_ltable='ltable_id', fk_rtable='rtable_id')
#  
# #       self.labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/I.csv'),  sep=',')
#      
#         # get features for feature vectors
#         match_f = self.get_feats(sample_A, sample_B)
#          
#         #from labeled data, get labeled fvs
#         self.labeled_dataset_seed = self.train_fvs(self.labeled_dataset_seed, match_f, i_attrs_after='gold_labels')
#          
#         self.context = {"dataset_a": sample_A, "dataset_b": sample_B }
#          
#         self.unlabeled_dataset = em.read_csv_metadata(os.path.join(os.path.dirname(__file__), 'data/unlabeled_dataset_fvs.csv'), key='_id',ltable=sample_A, rtable=sample_B, fk_ltable='ltable_id', fk_rtable='rtable_id')
#          
# #         self.unlabeled_dataset_seed = self.train_fvs(self.unlabeled_dataset_seed, match_f)
#                  
#         #create a model
#         self.model = RandomForestClassifier()
#         #create a labeler
#  
#         self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
#      
#         #create a selector
#         self.selector  = EntropyBasedExampleSelector()
#          
#  
#     #testing non batch mode
#     def test_active_learn_non_batch(self):
#         #create a learner
#         alearner = ActiveLearner(self.model, self.selector, self.labeler, 1, 10)
#         alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'ltable_id', 'rtable_id'], context=self.context, label_attr='gold_labels')
#         assert_equal(0,0)
#      
#     #testing batch mode
#     def test_active_learn_batch(self):
#         #create a learner
#         alearner = ActiveLearner(self.model, self.selector, self.labeler, 2, 2)
#         alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'ltable_id', 'rtable_id'], context=self.context, label_attr='gold_labels')
#         assert_equal(0,0)
#      
#     def test_active_learn_different_model(self):
#         #create a learner
#         alearner = ActiveLearner(self.model, self.selector, self.labeler, 2, 2)
#         alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'ltable_id', 'rtable_id'], context=self.context, label_attr='gold_labels')
#         assert_equal(0,0)    
#           