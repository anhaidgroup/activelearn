import operator
import numpy as np

from activelearn.exampleselector.uncertainty_based_example_selector import UncertaintySelector
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

class LeastConfidenceSelector(UncertaintySelector):
    
    def __init__(self):
        super(LeastConfidenceSelector, self).__init__()

    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
            """
            Used to select informative examples based on the confidence of the examples.
    
            Args:
                model (Model): Model that is used to compute the uncertainty measure of the example
                unlabeled_dataset (Pandas DataFrame): A Dataframe containing unlabeled examples
                exclude_attr (list): Attributes which are not feature attributes.
                batch_size (number): The number of examples to select
        
            Returns:
                The table of most informative examples to be labeled (DataFrame)
            """
            validate_input_table(unlabeled_dataset, 'unlabeled dataset')

            for attr in exclude_attrs:
                validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
                 
            feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)

            feature_values = unlabeled_dataset[feature_attrs]
            
            probabilities = model.predict_proba(feature_values)

            # compute the maximum probability of being classified into any class for the unlabeled pairs
            maxprobabilities = np.max(probabilities, axis=1)
            next_batch_idxs = np.argpartition(maxprobabilities, -batch_size)[:batch_size]
            return unlabeled_dataset.iloc[next_batch_idxs]
    