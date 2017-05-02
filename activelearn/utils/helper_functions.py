"""utilities"""

from activelearn.utils.validation import validate_attr

def remove_exclude_attr(feature_attrs, exclude_attrs, dataset):
    for attr in exclude_attrs:
        validate_attr(attr, dataset.columns, "attr", 'unlabeled_dataset')
                                
    if exclude_attrs:                                                       
        for attr in exclude_attrs:                                          
            feature_attrs.remove(attr) 
    
    return feature_attrs