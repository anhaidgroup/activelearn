from activelearn.labeler.labeler import Labeler
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_fn

class CliLabeler(Labeler):
    """
    A command line labeler for labeling raw instances
    
    Args:
        get_instruction_fn (Function): 	User provided function which specifies 
				       	the instruction to be displayed to
				       	the user before a particular example's
				        label is queried from the user.

        get_example_display_fn (Function): User provided function which 
					specifies how to fetch the raw 
					examples to be labeled.
					This function will take 2 arguments
					namely a feature vector and a context 
					object. The context is an object
					provided by the user that contains
					the necessary data to obtain the 
					raw representation of the feature
					vector to be labeled. This function
					will provide the necessary logic
					to obtain the raw representation of
					the examples to be labeled using the
					provided context.
					
		labels (Dictionary): A dictionary with the label text and values
		
    Attributes:
        get_instruction_fn (Function): An attribute to store the get_instruction_fn
        get_example_display_fn (Function): An attribute to store the get_example_display_fn
        labels: An attribute to store the labels dictionary
    """
    
    def __init__(self, get_instruction_fn, get_example_display_fn, labels):
        validate_fn(get_instruction_fn)
        validate_fn(get_example_display_fn)
        self.get_instruction_fn = get_instruction_fn
        self.get_example_display_fn = get_example_display_fn
        self.labels = labels
 
    def _input_from_stdin(self, banner_str):
        return raw_input(banner_str)
    
    def _default_get_instruction_fn(self):
        """
        Displays the default instruction message
        """
        return "Enter the Label"
    
    def _default_get_example_display_fn(self, example, context):
        """
        By default assume the example is the raw example 
        """
        return str(example)
    
    def validate_label_input(self, raw_label_str):
        if self.labels.has_key(raw_label_str):
            return True
        else:
            return False
    
    
    def label(self, examples_to_label, context, label_attr='label'):
        """
        Queries the labels of the instances to be queried from oracle
        
        Args:
            examples_to_label (Dataframe): 
            
            context (dictionary): A dictionary containing all the necessary
                                  context for the labeling function
            
            label_attr (string): The name of the attribute which contains the labels 
                                 of the instances
            
        Returns:
            A table of examples with labels annotated by the user (DataFrame)
        
        """
        # check if the input examples_to_label is a dataframe
        validate_input_table(examples_to_label, 'unlabeled dataset')
        
        #Show the instruction to the user 
        print(self.get_instruction_fn(context))
        
        user_labels = []
        
        for example in examples_to_label.iterrows():
            #Fetch and display the raw example to be labeled
            label_str = self._input_from_stdin(
                            self.get_example_display_fn(example, context))
            if self.validate_label_input(label_str):
                user_labels.append(self.labels[label_str])
            else:
                #Display error message if user enters a wrong label
                print("Incorrect Label. Pls try again")
                self.label(examples_to_label, context, label_attr='label')
            

        examples_to_label[label_attr] = user_labels
        return examples_to_label
