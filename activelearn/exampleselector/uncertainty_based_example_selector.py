
from activelearn.exampleselector.example_selector import ExampleSelector

class UncertaintySelector(ExampleSelector):
    """
    Base class for all uncertainty based example selectors
    """
    def __init__(self):
        super(UncertaintySelector, self).__init__()
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )
