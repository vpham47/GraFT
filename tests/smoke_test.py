"""End-to-end smoke test on tiny synthetic data (CPU, a few minutes).

    python tests/smoke_test.py            # everything
    python tests/smoke_test.py --quick    # skip training the additional backbones

Checks
  * main backbones: gates start at zero, the model equals its predictor at initialization,
    no-graph and fixed-gate variants, PatchTST patch settings
  * additional backbones: build, output shape, identity at initialization, backward pass
  * learning-rate schedule: warmup -> peak -> linear decay
  * presets: paper settings are loaded exactly, size classes follow N
  * data tools: CSV with timestamps -> dataset folder (time features derived correctly)
  * training: every main backbone learns (clearly beats persistence); every additional one trains
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run as runner  # noqa: E402
from graft.config import DEFAULTS  # noqa: E402
from graft.datasets import custom_dataset, size_class  # noqa: E402
from graft.registry import GRAFT_MODELS, MAIN_BACKBONES  # noqa: E402
from graft.presets import paper_configs, paper_preset, size_preset  # noqa: E402

NODES, STEPS_PER_DAY, SEQ_LEN, HORIZON = 12, 24, 12, 12

TINY = ['--device', 'cpu', '--graft_graph_dG', '8', '--graft_pred_latent', '16',
        '--graft_hidden', '16', '--graft_tod_dim', '4', '--graft_dow_dim', '4',
        '--graft_node_dim', '4', '--graft_num_layers', '2']
TRAIN = ['--bs', '8', '--max_epochs', '8', '--warmup_epochs', '1', '--warmup_init_lr', '0.0001',
         '--final_lr', '0.00005', '--lrate', '0.005', '--patience', '20', '--ema_decay', '0.9',
         '--graft_post_graph_ffn', '1']


def model_args(extra=()):
    args = dict(DEFAULTS)
    args.update({'graft_graph_dG': 8, 'graft_pred_latent': 16, 'graft_hidden': 16,
                 'graft_tod_dim': 4, 'graft_dow_dim': 4, 'graft_node_dim': 4,
                 'graft_num_layers': 2, 'graft_steps_per_day': STEPS_PER_DAY})
    args.update(dict(extra))
    return argparse.Namespace(**args)


def build(backbone, **extra):
    return GRAFT_MODELS[backbone](node_num=NODES, input_dim=3, output_dim=1, seq_len=SEQ_LEN,
                                  horizon=HORIZON, model_args=model_args(extra)).eval()


def make_batch(batch=3):
    x = torch.randn(batch, SEQ_LEN, NODES, 3)
    x[..., 1] = torch.rand(batch, SEQ_LEN, NODES)
    x[..., 2] = torch.randint(0, 7, (batch, SEQ_LEN, NODES)).float()
    return x


def predictor_only(model, x):
    emb = model._build_embeddings(x)
    h = torch.cat([x, emb], dim=-1)
    B, T, N, _ = h.shape
    y = model.predictor(h.permute(0, 2, 1, 3).reshape(B * N, T, -1))
    y = y.reshape(B, N, HORIZON, model.pred_latent).permute(0, 2, 1, 3)
    return model.out_proj(y) + x[:, -1:, :, :1]


def check_models():
    x = make_batch()
    for backbone in MAIN_BACKBONES:
        model = build(backbone, graft_post_graph_ffn=0)
        assert all(g == 0.0 for g in model.get_gate_magnitudes()['per_layer']), 'gates not zero at init'
        with torch.no_grad():
            out = model(x)
            assert out.shape == (3, HORIZON, NODES, 1), f'bad output shape {tuple(out.shape)}'
            assert torch.isfinite(out).all()
            assert torch.allclose(out, predictor_only(model, x), atol=1e-6), 'not identity at init'
        assert build(backbone, graft_num_layers=0).get_gate_magnitudes() == {'per_layer': [], 'mean': 0.0}
        fixed = build(backbone, graft_fixed_gate=1)
        assert all(g == 1.0 for g in fixed.get_gate_magnitudes()['per_layer'])
        assert not any('alpha' in n for n, _ in fixed.named_parameters()), 'fixed gate must not train'
        with torch.no_grad():
            assert torch.isfinite(build(backbone, graft_post_graph_ffn=1)(x)).all()
        print(f'PASS model      graft-{backbone}: zero gates, identity at init, variants')
    for length, stride in [(2, 1), (4, 2)]:
        model = build('patchtst', graft_patch_len=length, graft_patch_stride=stride)
        assert model.predictor.num_patches == (SEQ_LEN - length) // stride + 1
    print('PASS model      graft-patchtst: patch settings 2/1 and 4/2')


def check_additional():
    x = make_batch()
    for backbone in sorted(set(GRAFT_MODELS) - set(MAIN_BACKBONES)):
        model = build(backbone, graft_post_graph_ffn=0)
        assert all(g == 0.0 for g in model.get_gate_magnitudes()['per_layer']), 'gates not zero at init'
        with torch.no_grad():
            torch.manual_seed(0)
            out = model(x)
            torch.manual_seed(0)
            reference = predictor_only(model, x)
        assert out.shape == (3, HORIZON, NODES, 1) and torch.isfinite(out).all()
        assert torch.allclose(out, reference, atol=1e-5), f'{backbone}: not identity at init'
        model.train()
        model(x).sum().backward()
    print('PASS additional ' + ', '.join(sorted(set(GRAFT_MODELS) - set(MAIN_BACKBONES))) +
          ': build, identity at init, backward')


def check_scheduler():
    args = argparse.Namespace(max_epochs=20, warmup_epochs=4, lrate=0.002, warmup_init_lr=1e-4, final_lr=5e-5)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=args.lrate)
    sched = runner.build_scheduler(opt, args)
    lrs = []
    for _ in range(args.max_epochs):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert abs(lrs[0] - 1e-4) < 1e-9 and abs(lrs[4] - 2e-3) < 1e-9
    assert all(a >= b for a, b in zip(lrs[4:], lrs[5:])) and lrs[-1] >= 5e-5
    print('PASS scheduler  warmup -> peak -> linear decay')


def check_presets():
    configs = paper_configs()
    assert sorted(configs) == ['mlp', 'patchtst', 'tsmixer'] and all(len(v) == 14 for v in configs.values())
    sd = paper_preset('patchtst', 'sd')
    try:
        paper_preset('dlinear', 'sd')
        raise AssertionError('additional backbones have no paper preset')
    except ValueError:
        pass
    assert (sd['graft_graph_dG'], sd['graft_num_layers'], sd['lrate'], sd['bs']) == (128, 3, 0.001, 16)
    assert paper_preset('mlp', 'ca')['graft_num_layers'] == 1
    assert (size_class(170), size_class(716), size_class(8600)) == ('small', 'medium', 'large')
    spec = argparse.Namespace(num_nodes=4000, steps_per_day=96, tod_channel=1, dow_channel=2)
    cfg, size = size_preset('mlp', spec)
    assert size == 'large' and cfg['graft_num_layers'] == 1 and cfg['graft_hidden'] == 512
    print('PASS presets    paper settings (42 runs) loaded exactly; size classes follow N')


def sine_values(seed=0, days=21):
    rng = np.random.default_rng(seed)
    t = np.arange(days * STEPS_PER_DAY)
    phase = rng.uniform(0, 2 * np.pi, size=NODES)
    values = 50 + 10 * np.sin(2 * np.pi * t[:, None] / STEPS_PER_DAY + phase[None, :])
    return values + rng.normal(0, 1.0, size=values.shape)


def persistence_mae(data_dir):
    archive = np.load(os.path.join(data_dir, 'his.npz'))
    data, std = archive['data'], float(archive['std'])
    errors = [np.abs(data[t + 1:t + 1 + HORIZON, :, 0] - data[t, :, 0][None]).mean() * std
              for t in np.load(os.path.join(data_dir, 'idx_test.npy'))]
    return float(np.mean(errors))


def check_csv(tmp):
    import pandas as pd
    stamps = pd.date_range('2024-01-01 00:00', periods=21 * 24, freq='h')   # Monday 00:00, hourly
    frame = pd.DataFrame(sine_values(), columns=[f's{i}' for i in range(NODES)])
    frame.insert(0, 'timestamp', stamps.strftime('%Y-%m-%d %H:%M'))
    frame.iloc[5, 3] = np.nan
    csv = os.path.join(tmp, 'readings.csv')
    frame.to_csv(csv, index=False)
    out = os.path.join(tmp, 'from_csv')
    old = sys.argv
    sys.argv = ['csv_to_graft.py', '--csv', csv, '--out', out]
    try:
        sys.path.insert(0, os.path.join(ROOT, 'tools'))
        import csv_to_graft
        csv_to_graft.main()
    finally:
        sys.argv = old
    spec = custom_dataset(out)
    data = np.load(os.path.join(out, 'his.npz'))['data']
    assert spec.num_nodes == NODES and spec.steps_per_day == 24
    assert np.allclose(data[:24, 0, 1], np.arange(24) / 24), 'time of day not derived from timestamps'
    assert data[0, 0, 2] == 0 and np.isclose(data[24 * 6, 0, 2], 6 / 7), 'day of week not derived from timestamps'
    print('PASS csv        timestamps -> steps/day, time of day and day of week')
    return out


def train(argv):
    result = runner.main(argv)
    logging.shutdown()
    assert np.isfinite(result['mae']), f'non-finite MAE for {argv}'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='skip training the additional backbones')
    quick = parser.parse_args().quick

    check_scheduler()
    check_presets()
    check_models()
    check_additional()

    tmp = tempfile.mkdtemp(prefix='graft_smoke_')
    try:
        csv_dir = check_csv(tmp)
        baseline = persistence_mae(csv_dir)
        out = ['--output_dir', os.path.join(tmp, 'experiments')]
        for backbone in MAIN_BACKBONES:
            r = train(['--backbone', backbone, '--data', csv_dir] + TINY + TRAIN + out)
            assert r['mae'] < 0.5 * baseline, f'graft-{backbone}: MAE {r["mae"]:.3f} vs persistence {baseline:.3f}'
            assert len(r['gates']) == 2
            print(f'PASS training   graft-{backbone}: test MAE {r["mae"]:.3f} vs persistence {baseline:.3f}')

        if not quick:
            for backbone in sorted(set(GRAFT_MODELS) - set(MAIN_BACKBONES)):
                if backbone == 'timesnet':
                    continue    # far too slow to train on CPU; covered by check_additional
                r = train(['--backbone', backbone, '--data', csv_dir] + TINY + out +
                          ['--max_epochs', '2', '--warmup_epochs', '1', '--bs', '16', '--lrate', '0.005'])
                print(f'PASS training   graft-{backbone}: trains and evaluates (test MAE {r["mae"]:.3f}, 2 epochs)')
    finally:
        logging.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    print('ALL SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
