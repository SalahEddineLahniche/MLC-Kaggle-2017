import datetime as dt
import functools

'''
this function reverse a dictionary given the condition that both the keys and the values are unique
'''
def reverse_dict(d):
    return {d[key]: key for key in d}

'''
this decorator time the execution of a given function
'''

def formatted_now():
    import datetime as dt
    return dt.datetime.strftime(dt.datetime.today(), '%y%m%d-%H%M%S')
