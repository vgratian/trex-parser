
class Token():

    def __init__(self, index, form, l=None, p=None, xp=None, m=None, h=None, l1=None, l2=None, l3=None):
        self.index = int(index)
        self.form = form
        self.lemma = l
        self.pos = p if p not in ( ",", "''", ".", "``", ":" ) else 'PUNC'
        self.xpos = xp
        self.morph = m
        self.head = int(h) if h and h != '_' else '_'
        self.label = l1
        self.label2 = l2
        self.label3 = l3
    

    def __str__(self):
        return '\t'.join( str(x) for x in ( 
                            self.index, 
                            self.form, 
                            self.lemma, 
                            self.pos, 
                            self.xpos, 
                            self.morph, 
                            self.head, 
                            self.label, 
                            self.label2, 
                            self.label3 ) ) 