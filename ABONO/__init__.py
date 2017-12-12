import functools

import pandas as pd

from ABONO.Regressor import Regressor
from ABONO.Processer import Processer
from ABONO.Session import Session

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
def timed(session):
	def innertimed(f):
	    import time
	    @functools.wraps(f)
	    def wrapped(*args):
	        t=time.time() # get the current time
	        rslt = f(*args)
	        # print the current time - the recorded time (which is the elapsed time in seconds)
	        session.log("Time of execution is: {t:.0f} s".format(t=(time.time()) - t))
	        return rslt
	    return wrapped
	return innertimed

class model:
    def __init__(self, processer, session, offset=0, dcols=None, length=None, model=None, **kwargs):
        self.pr = processer
        self.session = session
        self.model = model
        self.offset = offset
        self.length = length
        self.dcols = dcols
        self.m_args = kwargs

    def run(self, cross_validate=False, processed_train_data=None, processed_test_data=None):
        if not processed_train_data:
            self.session.log("raw train dataset: {file}".format(file=TRAIN_PATH))
            self.session.init_train()
            tmp_train = self.session.get_train_filename()
            self.session.log("structured train dataset: {file}--".format(file=tmp_train))
            with open(TRAIN_PATH) as f:
                with open(tmp_train, 'w') as g:
                    self.pr.process(f, g, length=self.length, offset=self.offset)
            processed_train_data = tmp_train
        if not processed_test_data and not cross_validate:
            self.session.log("raw train dataset: {file}".format(file=TEST_PATH))
            self.session.init_test()
            tmp_test = self.session.get_test_filename()
            self.session.log("structured train dataset: {file}".format(file=tmp_test))
            with open(TEST_PATH) as f:
                with open(tmp_test, 'w') as g:
                    self.pr.process(f, g, length=self.length, offset=self.offset)
            processed_test_data = tmp_test
        self.df = pd.read_csv(processed_train_data)
        if not cross_validate:
            self.tdf = pd.read_csv(processed_test_data)
        else:
            self.tdf = None
        reg = Regressor(self.session, self.df, self.tdf, self.dcols, self.model, **self.m_args)
        if cross_validate:
            return reg.cross_validate()
        else:
            y = reg.predict()
            mse = reg.cross_validate(fit=False)
            return y, mse

        
