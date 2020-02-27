
import numpy as np


class FeatureModel():

    def __init__(self, corpus, window_size, objects=('stack', 'buffer'), token_types=('form', 'lemma', 'pos')):

        self.objects = objects
        self.token_types = token_types
        self.window_size = window_size
        self.hash_table = self.extract_hash_table(corpus)
        self.size = len( self.hash_table )
        print('Initialized Feature Model with {} dimensions'.format(self.size))


    def extract_hash_table(self, corpus):

        hash_table = {}

        for sentence in corpus:
            for token in sentence:

                form = token.form
                lemma = token.lemma
                pos = token.pos

                for obj in self.objects:
                    for position in range(self.window_size):

                        key1 = (obj, position, 'form', form)
                        key2 = (obj, position, 'lemma', lemma)
                        key3 = (obj, position, 'pos', pos)

                        if key1 not in hash_table:
                            hash_table[ key1 ] = len(hash_table)

                        if key2 not in hash_table:
                            hash_table[ key2 ] = len(hash_table)

                        if key3 not in hash_table:
                            hash_table[ key3 ] = len(hash_table)

        # Add None to each feature combination, this is for empty elements
        # TODO: maybe replace with special symbols

        for obj in self.objects:
            for position in range(self.window_size):
                for token_type in self.token_types:
                    key = (obj, position, token_type, None)
                    hash_table[key] = len(hash_table)

        # Finally add BIAS
        hash_table['<BIAS>'] = len(hash_table)
                    
        return hash_table


    def extract(self, configuration):

        #features = np.zeros( shape=(1, self.size) )
        features = []

        elements = self.get_top_elements(configuration)
        tokens = configuration.sentence.tokens

        for obj in elements:
            for token_index, position in zip(elements[obj], range(self.window_size)):

                # token indices are None if element in buffer/stack is empty
                # in that case our feature should be None as well
                if token_index is None:
                    form = None
                    lemma = None
                    pos = None
                else:
                    token = tokens[token_index]
                    form = token.form
                    lemma = token.lemma
                    pos = token.pos

                #print(obj, '<', position, '(', token_index, ') => [', form, lemma, pos, ']')

                # If feature is in our hash table, we want to add it to the feature vector
                # Otherwise we ignore features that are not in our dimensions
                try:    
                    feature_index = self.hash_table[ (obj, position, 'form', form) ]
                    #features[0][feature_index] = 1.0
                    features.append( feature_index )
                except KeyError:
                    pass
                try:    
                    feature_index = self.hash_table[ (obj, position, 'lemma', lemma) ]
                    #features[0][feature_index] = 1.0
                    features.append( feature_index )
                except KeyError:
                    pass
                try:    
                    feature_index = self.hash_table[ (obj, position, 'pos', pos) ]
                    #features[0][feature_index] = 1.0
                    features.append( feature_index )
                except KeyError:
                    pass    

        # Set BIAS to one
        #features[0][-1] = 1.0
        features.append( self.hash_table['<BIAS>'] )

        return np.array( features, dtype=int )


    def get_top_elements(self, configuration):

        indices = { 'stack' : [], 'buffer': [] }

        i = min( self.window_size, len(configuration.stack) )
        indices['stack'] = [None] * (self.window_size-i) + configuration.stack[-i:]

        i = min( self.window_size, len(configuration.buffer) )
        indices['buffer'] = [None] * (self.window_size-i) + configuration.buffer[-i:]

        return indices
        
