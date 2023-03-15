from minillm.engine.modules import QuantLinear

def make_quant(module, names, bits, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, QuantLinear(bits, tmp.in_features, tmp.out_features)
            )
    for name1, child in module.named_children():
        make_quant(child, names, bits, name + '.' + name1 if name != '' else name1)
