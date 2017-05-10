 
import operator
import numpy as np
 
from activelearn.exampleselector.uncertainty_based_example_selector import UncertaintySelector
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr
from six.moves import xrange

class SmallestMarginSelector(UncertaintySelector):
     
    def __init__(self):
        super(SmallestMarginSelector, self).__init__()
     
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
     
    def _utility_example(self, probability):
        return self._compute_margin(probability)
     
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
        """
        Used to select informative examples based on the margin.

        Args:
            model (Model): Model that is used to compute the uncertainty measure of the example
            unlabeled_dataset (DataFrame): A DataFrame containing unlabeled examples
            exclude_attr (list): Attributes which are not feature attributes.
            batch_size (number): The number of examples to select
        
        Returns:
                The table of most informative examples to be labeled (DataFrame) 
        """
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        #validate exclude attr
        for attr in exclude_attrs:
            validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
        
        # remove exclude attrs                                                  
        
        feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)
        
        feature_values = unlabeled_dataset[feature_attrs]
            
        probabilities = model.predict_proba(feature_values)

        # compute the entropy for the unlabeled pairs
        margins = {}
        for i in xrange(len(probabilities)):
            margins[i] = self._compute_margin(probabilities[i])
        # compute the margin of uncertainity for the unlabeled pairs
        candidate_examples = sorted(margins.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(margins))]
        next_batch_idxs = [val[0] for val in candidate_examples]
        return unlabeled_dataset.iloc[next_batch_idxs]
     
