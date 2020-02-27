

def Shift(configuration, label=None, create_copy=False):
    c = configuration.copy() if create_copy else configuration
    c.stack.append( c.buffer.pop() )
    #c.history.append( (Shift, label ) )
    c.history.append( Shift )
    return c


def LeftArc(configuration, label=None, create_copy=False):
    c = configuration.copy() if create_copy else configuration
    child, head = c.stack[-2:]
    c.stack.remove(child)

    # If we reverse engineer transitions, we take label from gold arcs
    # If argument label is not provided, we assume that's the case
    # leaving it out as long as with do only UAS
    #dp_label = label or c.gold_arcs.get( (head, child), None )
    
    #print('LeftARc=', dp_label)
    
    #c.parsed_arcs[ (head, child )] = dp_label
    #c.history.append( (LeftArc, dp_label ) )

    c.parsed_arcs.append( (head, child) )
    c.history.append(LeftArc)
    return c


def RightArc(configuration, label=None, create_copy=False):
    c = configuration.copy() if create_copy else configuration
    child = c.stack.pop()
    head = c.stack[-1]
    #dp_label = label or c.gold_arcs.get( (head, child), None )

    #print('LeftARc=', dp_label)

    #c.parsed_arcs[ (head, child )] = dp_label
    #c.history.append( (RightArc, dp_label ) )

    c.parsed_arcs.append( (head, child) )
    c.history.append(RightArc)

    return c