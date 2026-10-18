"""Synthetic native-model regression, no pretrained data or quality metrics."""
import pytest
torch=pytest.importorskip('torch')
transformers=pytest.importorskip('transformers')
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.bridges.pre_block import PreBlockIntervention


@pytest.mark.parametrize('encoder',['videomae','timesformer'])
@pytest.mark.parametrize('method',['attention_topk','visionzip_internal','uniform'])
@pytest.mark.parametrize('ratio',[1.,.5,.3])
def test_native_identity_true_pre_block_shapes_and_repeated_forward(encoder,method,ratio):
    torch.manual_seed(0)
    if encoder=='videomae':
        cfg=transformers.VideoMAEConfig(image_size=8,patch_size=2,num_frames=4,tubelet_size=2,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32)
        model=transformers.VideoMAEModel(cfg).eval()
        native,after={1.:32,.5:16,.3:10}[1.],{1.:32,.5:16,.3:10}[ratio]
    else:
        cfg=transformers.TimesformerConfig(image_size=8,patch_size=2,num_frames=4,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,attention_type='divided_space_time',drop_path_rate=0.)
        model=transformers.TimesformerModel(cfg).eval();native=65;after={1.:65,.5:33,.3:21}[ratio]
    bridge=create_observation_bridge(encoder,model)
    try:context=PreBlockIntervention(bridge,1,method,ratio)
    except ValueError as exc:
        if str(exc).startswith('unsupported native'):pytest.skip('local Transformers differs from SHA-audited server: '+str(exc))
        raise
    x=torch.randn(2,4,3,8,8)
    with torch.no_grad():
        dense=model(x).last_hidden_state
        with context:
            for _ in range(2):
                output=model(x).last_hidden_state;r=context.receipt()
                assert r['before_tokens']==native and r['after_tokens']==after
                assert r['inputs'][1][1]==after and r['mlp_inputs'][0][1]==native
                assert r['signal_block']==1 and r['target_block']==2
                if ratio==1:torch.testing.assert_close(output,dense,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(model(x).last_hidden_state,dense,rtol=0,atol=0)
    assert not hasattr(model,'_vadbench_pre_block')
    assert all('forward' not in b.__dict__ for b in bridge._blocks)


def test_invalid_target_does_not_modify_model():
    cfg=transformers.VideoMAEConfig(image_size=8,patch_size=2,num_frames=4,tubelet_size=2,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32)
    model=transformers.VideoMAEModel(cfg).eval();bridge=create_observation_bridge('videomae',model)
    with pytest.raises(ValueError,match='previous'):PreBlockIntervention(bridge,0,'uniform',.5)
    assert not hasattr(model,'_vadbench_pre_block')
