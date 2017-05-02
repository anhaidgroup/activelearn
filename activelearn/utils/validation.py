"""Validation utilities"""

import pandas as pd


def validate_input_table(table, table_label):
    """Check if the input table is a dataframe."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(table_label + ' is not a dataframe')
    return True


def validate_attr(attr, table_cols, attr_label, table_label):
    """Check if the attribute exists in the table."""
    if attr not in table_cols:
        raise AssertionError(attr_label + ' \'' + attr + '\' not found in ' + \
                             table_label) 
    return True

def validate_fn(attr):
    """Check if the attribute is a function"""
    if not callable(attr):
        raise TypeError(attr + ' is not a function')
    return True