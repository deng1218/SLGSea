"""Structured SAR prompt assembly."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'prompts/templates.json').read_text(encoding='utf-8'))
CLASSES=('ship','sea','land')
ATTRS=('scale','topology','scattering')

def render(values):
    attr={c:{a:SPEC['attribute_templates'][c][a].format(value=values[c][a]) for a in ATTRS} for c in CLASSES}
    return {'scale_topology':[attr[c][a] for c in CLASSES for a in ('scale','topology')],
            'integrated':[SPEC['integrated_templates'][c].format(**values[c]) for c in CLASSES],
            'scattering':[attr[c]['scattering'] for c in CLASSES],
            'sqc':list(SPEC['sqc_reference_prompts'])}

def fixed_prompts():
    return render(SPEC['fixed_inference_descriptors'])

def training_prompts(row):
    return render(recovered_descriptors(row))

def recovered_descriptors(row):
    """Combine quantity, size, spatial relations, and density descriptors."""
    ship,sea,land=(row['T_'+c] for c in CLASSES)
    interference={'low interference':'weak scattering interference','medium interference':'moderate scattering interference','high interference':'strong scattering interference','no-ship':'no ship-associated scattering'}
    sea_state={'calm':'low backscatter variation','moderate':'moderate backscatter variation','rough':'strong backscatter variation','no-sea':'no sea-surface scattering','unknown':'unknown backscatter variation'}
    land_state={'sparse-bright':'sparse bright scatterers','dense-bright':'dense bright scatterers','clustered-bright':'clustered bright scatterers','no-land':'no land scattering'}
    count={'none':'no','few':'few','medium':'a moderate number of','many':'many'}[ship['scale']]
    sizes={'none':'','mostly-small':'mostly small','mostly-large':'mostly large','mixed-size':'mixed-size'}[ship['size_composition']]
    topo={'nearshore':'distributed near the shoreline','offshore':'distributed offshore','no-ship':'absent'}[ship['topology']]
    density={'isolated':'as isolated targets','moderate':'with moderate clustering','clustered':'in clustered groups','no-ship':''}[ship['density']]
    coverage=lambda c,v: {'no-'+c:'absent',c+'-limited':'limited in spatial extent','balanced':'balanced in spatial extent',c+'-dominant':'dominant in spatial extent'}[v]
    values={
      'ship':{'scale':(count+' '+sizes).strip(),'topology':(topo+' '+density).strip(),'scattering':interference[ship['state']]},
      'sea':{'scale':coverage('sea',sea['scale']),'topology':sea['topology'],'scattering':sea_state[sea['state']]},
      'land':{'scale':coverage('land',land['scale']),'topology':land['topology']+' '+land['ship_relation'],'scattering':land_state[land['state']]}}
    # Preserve an explicit absence rather than treating absence as an omitted attribute.
    for c in ['sea','land']:
        if row['T_'+c]['scale']=='no-'+c: values[c]['topology']='no '+c+' regions'
    return values
