def flatten_list(l):
    '''
    Flattens a list of lists
    '''
    return [item for sublist in l for item in sublist]