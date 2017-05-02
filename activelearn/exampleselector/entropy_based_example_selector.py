
import operator
import pandas as pd

from activelearn.exampleselector.uncertainty_based_example_selector import UncertaintyBasedExampleSelector

from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr
from six.moves import xrange

class EntropyBasedExampleSelector(UncertaintyBasedExampleSelector):
    """
    Entropy based Uncertainty example selection
    """
    def __init__(self):
        super(EntropyBasedExampleSelector, self).__init__()
    
    
    def _compute_entropy(self, probability):
        if 0 in probability:
            return 0
        else:
            return pd.np.sum(-probability * pd.np.log(probability))
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, 
                        batch_size=1):
        """
        Used to select informative examples based on the entropy of the examples.

        Args:
            model (Model): Model that is used to compute the uncertainty measure of the example
            unlabeled_dataset (Pandas DataFrame): A DataFrame containing unlabeled examples
            exclude_attrs (list): Attributes which are not feature attributes.
            batch_size (number): The number of examples to select
        
        Returns:
            The table of most informative examples to be labeled (DataFrame)
        """
        # check if the input candset is a dataframe
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        
        #validate exclude attr
        for attr in exclude_attrs:
            validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
        
        # remove exclude attrs
        feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)

        feature_values = unlabeled_dataset[feature_attrs]
        
        # compute the prediction probabilities for the unlabeled dataset
        probabilities = model.predict_proba(feature_values) 

        entropies = {} 
        # compute the entropy for the unlabeled pairs
        for i in xrange(len(probabilities)):
            entropies[i] = self._compute_entropy(probabilities[i])
        
        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]

        next_batch_idxs = [val[0] for val in candidate_examples]
        return unlabeled_dataset.iloc[next_batch_idxs]
