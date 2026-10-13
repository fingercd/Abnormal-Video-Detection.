import copy
import pytest
from scripts.icassp2027.promote_relation_pipeline import validate


def fixture():
    config={'workers':1,'prefetch':2,'deduplicate':True}
    acceptance={'status':'accepted','pixel_exact':True,'cpu_four_stream_speedup':1.5,
        'gpu_median_speedup':1.3,'configuration':config,'cases':[
            {'method':m,'pooled_exact':True,'speedup':1.2}
            for m in ('identity','attention_topk','visionzip_internal','uniform')]}
    cloud={'verified':True,'state':'FINISHED','completion_summary':{'qa/completed':{'value':1}}}
    return acceptance,cloud,copy.deepcopy(config)


def test_promotion_passes_all_gates():
    validate(*fixture())


@pytest.mark.parametrize('failure',['pixels','output','cpu','gpu','regression','missing_method','cloud','summary','config'])
def test_promotion_refuses_failed_gate(failure):
    a,c,config=fixture()
    if failure=='pixels':a['pixel_exact']=False
    if failure=='output':a['cases'][0]['pooled_exact']=False
    if failure=='cpu':a['cpu_four_stream_speedup']=1
    if failure=='gpu':a['gpu_median_speedup']=1
    if failure=='regression':a['cases'][0]['speedup']=.9
    if failure=='missing_method':a['cases'].pop()
    if failure=='cloud':c['state']='CRASHED'
    if failure=='summary':c['completion_summary']={}
    if failure=='config':config['workers']=2
    with pytest.raises(ValueError):validate(a,c,config)


def numerical_fixture():
    a,c,config=fixture()
    vision=a['cases'][2]
    vision.update(pooled_exact=False,native_repeat_max_abs=2e-6,native_optimized_max_abs=3e-6,
        relative_l2_max=4e-7,numerical_policy='native_repeatability_bound_v1')
    return a,c,config


def test_native_repeat_bounded_visionzip_only():
    validate(*numerical_fixture())


@pytest.mark.parametrize('field,value',[
    ('native_repeat_max_abs',0),('native_repeat_max_abs',2e-5),
    ('native_optimized_max_abs',5e-6),('relative_l2_max',2e-6),
    ('numerical_policy','unchecked')])
def test_visionzip_repeatability_cannot_relax_arbitrary_difference(field,value):
    a,c,config=numerical_fixture();a['cases'][2][field]=value
    with pytest.raises(ValueError):validate(a,c,config)
