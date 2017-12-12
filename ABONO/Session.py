import os
import datetime as dt

class Session:
    def __init__(self, debug=True):
        self.dir = dt.datetime.strftime(dt.datetime.today(), '%y%m%d-%H%M%S')
        os.mkdir('data/{d}'.format(d=self.dir))
        self.strlog = 'log.txt'
        self.rslts = 'rslts.txt'
        self.train = 'train.csv'
        self.test = 'test.csv'
        self.model = 'model.dat'
        self.debug = debug
        self.log_format = '[{date}] {msg}'

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, type, value, traceback):
        if self.logf:
            self.logf.close()
        if self.rsltsf:
            self.rsltsf.close()
        if self.trainf:
            self.trainf.close()
        if self.testf:
            self.logf.close()
        

    def get_log_filename(self):
        return 'data/{d}/{s}'.format(d=self.dir, s=self.strlog)

    def get_results_filename(self):
        return 'data/{d}/{s}'.format(d=self.dir, s=self.rslts)

    def get_train_filename(self):
        return 'data/{d}/{s}'.format(d=self.dir, s=self.train)

    def get_test_filename(self):
        return 'data/{d}/{s}'.format(d=self.dir, s=self.test)

    def get_model_filename(self):
        return 'data/{d}/{s}'.format(d=self.dir, s=self.model)

    def init(self):
        self.logf = open(self.get_log_filename(), 'w')
        self.rsltsf = open(self.get_results_filename(), 'w')
        self.trainf = None
        self.testf = None
        self.modelf = None

    def init_train(self):
        self.trainf = open(self.get_train_filename(), 'w')

    def init_test(self):
        self.testf = open(self.get_test_filename(), 'w')

    def init_model(self):
        self.modelf = open(self.get_model_filename(), 'wb')


    def log(self, msg, rslts=False):
        t = dt.datetime.strftime(dt.datetime.today(), '%y/%m/%d - %H:%M:%S')
        if not rslts:
            print(self.log_format.format(msg=msg, date=t), file=self.logf)
        else:
            print(self.log_format.format(msg=msg, date=t), file=self.rsltsf)
        if self.debug:
            print(self.log_format.format(msg=msg, date=t))
