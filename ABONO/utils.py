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
def timed(f):
    import time
    @functools.wraps(f)
    def wrapped(*args):
        t=time.time() # get the current time
        f(*args)
        # print the current time - the recorded time (which is the elapsed time in seconds)
        print("Time of execution is: {t:.0f} s".format(t=(time.time()) - t))
    return wrapped

def formatted_now():
    import datetime as dt
    return dt.datetime.strftime(dt.datetime.today(), '%y%m%d-%H%M%S')
