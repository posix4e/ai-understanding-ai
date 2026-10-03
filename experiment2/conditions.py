"""All E2 interventions specified before discovery; no head selected by outcome."""
import json
from itertools import combinations
from pathlib import Path

def op(layer,heads,kind,**extra):
    return {'layer':layer,'component':'z','positions':[1,3,5,7] if layer==0 else [8],
            'heads':list(heads),'kind':kind,**extra}

def build():
    discovery=[{'id':'clean','group':'reference','ops':[]}]
    novel=[]
    for layer in range(2):
        label=f'L{layer+1}'
        for kind in ['mean','zero','resample']:
            for head in range(4):
                discovery.append({'id':f'{label}_{kind}_h{head}','group':'familiar',
                                  'ops':[op(layer,[head],kind)]})
            discovery.append({'id':f'{label}_{kind}_all','group':'familiar','ops':[op(layer,range(4),kind)]})
            for heads in combinations(range(4),2):
                novel.append({'id':f'{label}_{kind}_pair'+''.join(map(str,heads)),
                              'group':'pair','ops':[op(layer,heads,kind)]})
            for head in range(4):
                novel.append({'id':f'{label}_{kind}_rescue{head}', 'group':'triple_rescue',
                              'ops':[op(layer,[head],'rescue',corruption=kind)]})
            wrong=op(layer,range(4),kind)
            wrong['positions']=[8] if layer==0 else [1,3,5,7]
            discovery.append({'id':f'{label}_{kind}_wrong_position','group':'control','ops':[wrong]})
        for heads in [[0],[1],[2],[3],list(range(4))]:
            suffix='all' if len(heads)==4 else f'h{heads[0]}'
            discovery.append({'id':f'{label}_matched_{suffix}','group':'control',
                              'ops':[op(layer,heads,'resample',donor='matched_donor_tokens')]})
        discovery.append({'id':f'{label}_noop','group':'control','ops':[op(layer,range(4),'noop')]})
        for kind in ['zero','mean']:
            discovery.append({'id':f'{label}_mlp_{kind}','group':'control',
                'ops':[{'layer':layer,'component':'mlp_out','positions':[1,3,5,7] if layer==0 else [8],'kind':kind}]})
    for kind in ['mean','resample']:
        for h1 in range(4):
            for h2 in range(4):
                novel.append({'id':f'cross_{kind}_{h1}{h2}','group':'cross_layer',
                              'ops':[op(0,[h1],kind),op(1,[h2],kind)]})
    return {'discovery':discovery,'novel':novel,'confirmatory':discovery+novel}

if __name__=='__main__':
    result=build()
    Path('experiment2/conditions.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:len(v) for k,v in result.items()})
