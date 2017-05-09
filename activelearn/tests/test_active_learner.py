import unittest
from mock import MagicMock
from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from activelearn.exampleselector.entropy_selector import EntropySelector
from activelearn.labeler.cli_labeler import CliLabeler
from activelearn.activelearner.active_learner import ActiveLearner
from activelearn.utils.validation import validate_attr

class ActiveLearnerTests(unittest.TestCase):
    
    def sample_get_instruction_fn(self, context):
        return "Can you enter the label y or n?"

    def get_example_display_fn(self, example, context):
        table_a = context["dataset_a"]
        table_b = context["dataset_b"]
        example_A = table_a[table_a["A.ID"] == example["l_ID"]]
        example_B = table_b[table_b["B.ID"] == example["r_ID"]]
        return str(example_A) + "\n" + str(example_B)

    def setUp(self):
        dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/table_A.csv')).head(1000)
        dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/table_B.csv')).head(1000)
        self.context = {"dataset_a": dataset_a, "dataset_b": dataset_b }
        # labeled data, typically small in number in dataFrame format
        self.labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/seed.csv'),  sep='\t')
    
        self.unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/sample_fvs.csv'), sep='\t')
        


    def test_active_learn_non_batch(self):
        """testing non batch active learn loop for 2 iterations"""
        #create a model
        model = RandomForestClassifier() 
        #create a labeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
        
        label_attr = 'label'
        
        #mock user labels
        user_labels = [[0],[1]]
        
        #create mock labeled data
        gold_labeled_data1 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[0]])
        gold_labeled_data1[label_attr] = user_labels[0]
        gold_labeled_data2 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[1]])
        gold_labeled_data2[label_attr] = user_labels[1]
        
        #Mock the labeler to return gold data
        self.labeler.label = MagicMock()
        self.labeler.label.side_effect = [gold_labeled_data1, gold_labeled_data2]
        #create a selector
        self.selector  = EntropySelector()
        #create a learner
        alearner = ActiveLearner(model, self.selector, self.labeler, 1, 2)
        
        #length of unlabeled dataset
        len_unlabeled_data_before_learning = len(self.unlabeled_dataset)
        
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
        
        len_unlabeled_data_after_learning = len(self.unlabeled_dataset)
        #after two iterations we expect the length of unlabeled datset to have reduced by two
        assert_equal(2,len_unlabeled_data_before_learning - len_unlabeled_data_after_learning)
    
    def test_active_learn_batch(self):
        """testing batch mode active learn loop for 2 iterations"""
        #create a model
        model = RandomForestClassifier()
        #create a labeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
        
        label_attr = 'label'
        
        #mock user labels
        user_labels = [[0,1],[0,1]]
        
        #create mock labeled data
        gold_labeled_data1 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[0,1]])
        gold_labeled_data1[label_attr] = user_labels[0]
        gold_labeled_data2 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[2,3]])
        gold_labeled_data2[label_attr] = user_labels[1]
        
        #Mock the labeler to return gold data
        self.labeler.label = MagicMock()
        self.labeler.label.side_effect = [gold_labeled_data1, gold_labeled_data2]
        #create a selector
        self.selector  = EntropySelector()
        #create a learner
        alearner = ActiveLearner(model, self.selector, self.labeler, 2, 2)
        
        #length of unlabeled dataset
        len_unlabeled_data_before_learning = len(self.unlabeled_dataset)
        
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')

        #length of unlabeled dataset
        len_unlabeled_data_after_learning = len(self.unlabeled_dataset)
        
        #after two iterations we expect the length of unlabeled datset to have reduced by two
        assert_equal(4,len_unlabeled_data_before_learning - len_unlabeled_data_after_learning)  
#     
#     #test that in batch mode the loop exits prematurely with suitable error if the number of examples to select is exhausted
#    
    def test_batch_active_learn_error(self):
        """Testing with a different model"""
        #create a model
        model = SVC(probability=True)
        #create a labeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
        label_attr = 'label'
        
        #mock user labels
        user_labels = [[0,0,0,1],[1,0,0,0], [1,0]]
        
        #create mock labeled data
        gold_labeled_data1 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[0,1,2,3]])
        gold_labeled_data1[label_attr] = user_labels[0]
        gold_labeled_data2 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[4,5,6,7]])
        gold_labeled_data2[label_attr] = user_labels[1]
        gold_labeled_data3 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[8,9]])
        gold_labeled_data3[label_attr] = user_labels[2]
        
        #Mock the labeler to return gold data
        self.labeler.label = MagicMock()
        self.labeler.label.side_effect = [gold_labeled_data1, gold_labeled_data2, gold_labeled_data3]
        
        #create a selector
        self.selector  = EntropySelector()
        
        #length of unlabeled dataset
        len_unlabeled_data_before_learning = len(self.unlabeled_dataset)
        
        #create a learner
        alearner = ActiveLearner(model, self.selector, self.labeler, 4, 3)
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
        
        #length of unlabeled dataset
        len_unlabeled_data_after_learning = len(self.unlabeled_dataset)
        
        assert_equal(10,len_unlabeled_data_before_learning - len_unlabeled_data_after_learning) 
          
    def test_active_learn_different_model(self):
        """Testing with a different model"""
        #create a model
        model = SVC(probability=True)
        #create a labeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
        label_attr = 'label'
        
        #mock user labels
        user_labels = [[0],[1]]
        
        #create mock labeled data
        gold_labeled_data1 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[0]])
        gold_labeled_data1[label_attr] = user_labels[0]
        gold_labeled_data2 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[1]])
        gold_labeled_data2[label_attr] = user_labels[1]
        
        #Mock the labeler to return gold data
        self.labeler.label = MagicMock()
        self.labeler.label.side_effect = [gold_labeled_data1, gold_labeled_data2]
        
        #create a selector
        self.selector  = EntropySelector()
        #create a learner
        alearner = ActiveLearner(model, self.selector, self.labeler, 2, 2)
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
    
    
    #Testing with wrong labeler
    @raises(TypeError)
    def test_active_learn_with_wrong_labeler(self):
        
        #create a model
        model = SVC(probability=True)
        #create a labeler
        self.labeler = {}
        
        label_attr = 'label'
        
        #create a selector
        self.selector  = EntropySelector()
        #create a learner
        alearner = ActiveLearner(model, self.selector, self.labeler, 2, 2)
        
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
 
    
    #Testing with wrong selector
    @raises(TypeError)
    def test_active_learn_with_wrong_example_selector(self):
        
        #create a model
        model = SVC(probability=True)
        #create a labeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
        label_attr = 'label'
        
        #mock user labels
        user_labels = [[0],[1]]
        
        #create mock labeled data
        gold_labeled_data1 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[0]])
        gold_labeled_data1[label_attr] = user_labels[0]
        gold_labeled_data2 = pd.DataFrame.copy(self.unlabeled_dataset.iloc[[1]])
        gold_labeled_data2[label_attr] = user_labels[1]
        
        
        #Mock the labeler to return gold data
        self.labeler.label = MagicMock()
        self.labeler.label.side_effect = [gold_labeled_data1, gold_labeled_data2]
        label_attr = 'label'
        
        #create a dummy selector
        selector  = {}
        #create a learner
        alearner = ActiveLearner(model, selector, self.labeler, 2, 2)
        
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')