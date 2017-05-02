
from activelearn.exampleselector.example_selector import ExampleSelector

class UncertaintyBasedExampleSelector(ExampleSelector):
    """
    Base class for all uncertainty based example selectors
    """
    def __init__(self):
        super(UncertaintyBasedExampleSelector, self).__init__()
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )
