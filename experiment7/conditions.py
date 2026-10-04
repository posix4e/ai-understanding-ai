"""Fixed E7 discovery grid. All additions are explicitly oracle diagnostics."""
import torch

CONDITIONS = [dict(id='native',grouped=False,patch=False,mask='physical',guard=False),
              dict(id='grouped',grouped=True,patch=False,mask='physical',guard=False),
              dict(id='grouped_noop',grouped=True,patch=False,mask='noop',guard=False)]
for graph in ('physical','delete','add','transport'):
    for patch in (False,True):
        CONDITIONS.append(dict(id=('keypatch_' if patch else 'guard_')+graph,
            grouped=True,patch=patch,mask=graph,guard=True))
CONDITIONS.append(dict(id='transport_all',grouped=True,patch=False,mask='transport_all',guard=False))


def masks(encoded):
    from experiment6.conditions import construct_mask
    p=encoded.permutation(True)
    physical=torch.ones(len(p),len(p),dtype=torch.bool).tril()
    transport=p[None,:]<=p[:,None]
    result=dict(physical=physical,transport=transport,delete=physical&transport,
                add=physical|transport,guard=construct_mask(encoded,'guard_block0'))
    assert int((transport&~physical).sum())==24
    assert int((physical&~transport).sum())==24
    assert torch.equal(result['delete'],result['guard'])
    return result


def layer_masks(condition,available):
    if condition['mask']=='transport_all': return {k:available['transport'] for k in range(12)}
    if condition['mask']=='noop': return {k:available['physical'] for k in range(12)}
    result={0:available['guard']} if condition['guard'] else {}
    if condition['mask']!='physical':
        result.update({k:available[condition['mask']] for k in range(1,12)})
    return result
