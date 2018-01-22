import re

CSV_REGEXP = re.compile(r"(?:(?<=,)|(?<=^))(\"(?:[^\"]|\"\")*\"|[^,]*)")

class Processer:
    def __init__(self, session, new_cols=[], mapper={}, drop_cols=[], convoluted_mappers=[]):
        self.mapper = mapper
        self.convoluted_mappers = convoluted_mappers
        self.new_cols = new_cols
        self.session = session
        self.drop_cols = drop_cols

    '''
    return the different csv elements using the regular expression csv_regexp
    '''        
    def parse_line(self, line):
        return map(str, CSV_REGEXP.findall(line))

    def process(self, f, g, length=None, offset=0):
        head = next(f)
        cols = list(map(str.strip, head.split(',')))

        new_head = cols + self.new_cols
        for col in self.drop_cols:
            new_head.remove(col)

        g.write(','.join(new_head) + '\n')
        self.session.log('Processing...')
        
        index=0
        for line in f:
            if index < offset:
                continue
            
            columns = map(str.strip, head.split(','))
            objs = map(float, self.parse_line(line))
            objs = {k: v for k, v in zip(columns, objs)}

            new_cols = {k: self.mapper.get(k, lambda x: 0)(objs) for k in self.new_cols}

            for m in self.convoluted_mappers:
                a = m(objs)
                # print(m)
                new_cols.update(a)

            objs = {**objs, **new_cols}

            for col in self.drop_cols:
                objs.pop(col, None)

            str_objs = [str(objs[k]) for k in new_head]
            g.write(','.join(str_objs) + "\n")
            
            self.session.log('Line {index} is processed'.format(index=index))
            index += 1
            if length:
                if index >= offset + length:
                    break
        self.session.log('Finished processing !')
