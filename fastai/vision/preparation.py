
from pathlib import Path

PATH = '/home/domas/.fastai/data/FlickrLogos_47'

def get_nr2classname_dict(path=PATH): # str -> str
    lines = Path(path, 'className2ClassID.txt').open().readlines()
    d = {int(l.split('\t')[1][:-1])+1:l.split('\t')[0] for l in lines}
    d[0] = 'no-logo'
    return d


class Logo():
    def __init__(self):
        self.path = Path('') # path to img
        self.test = False # test set
        self.logo = True # is there logo
        self.ann = [] # annotations


class Annotation():
    def __init__(self):
        self.bb = [0,0,0,0] # bounding box
        self.mask = Path('') # path to mask
        self.c = '' # class name
        self.difficult = False # is difficult


def get_logos(path=PATH):

    nr2c = get_nr2classname_dict(path)
    out = []
    
    for fld in ['train', 'test']:
        lines = Path(path, fld, 'filelist.txt').open().readlines()
        
        for l in lines:
            im = Path(path, fld, l[:-1])
            
            lg = Logo()
            lg.path = im
            lg.test = True if fld == 'test' else False
            
            if str(im).split('/')[-2] == 'no-logo':
                lg.logo = False
                out.append(lg)
                continue
            
            annlines = Path(str(im)[:-4] + '.gt_data.txt').open().readlines()
            
            for al in annlines:
                b = al.split(' ')
                
                a = Annotation()
                for i in range(4):
                    a.bb[i] = int(b[i])
                    
                a.c = nr2c[ int(b[4])+1 ]
                a.mask = Path('{}.{}.png'.format(str(im)[:-4], b[6]))
                a.difficult = True if b[7] == '1' else False
                
                lg.ann.append(a)
                
            out.append(lg)
    return out