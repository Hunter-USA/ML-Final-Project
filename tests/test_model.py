import numpy as np
import torch

from sharp.features import FeatureStats
from sharp.labels import N_CLASSES
from sharp.model import StructureNet, count_parameters, load_checkpoint, run_model, save_checkpoint


def _model():
    torch.manual_seed(0)
    return StructureNet(n_bands=64, conv_channels=(8, 16, 16), gru_hidden=32, gru_layers=2).eval()


def test_output_shapes():
    m = _model()
    x = torch.randn(3, 120, 64)
    b, s = m(x, torch.tensor([120, 90, 50]))
    assert b.shape == (3, 120)
    assert s.shape == (3, 120, N_CLASSES)
    assert count_parameters(m) > 0


def test_padding_does_not_change_predictions():
    """A padded song's outputs must match running it alone (BiGRU reverses within each length)."""
    m = _model()
    x = torch.randn(2, 100, 64)
    x[1, 60:] = 0.0
    with torch.no_grad():
        b_batch, s_batch = m(x, torch.tensor([100, 60]))
        b_alone, s_alone = m(x[1:2, :60])
    # The CNN's last ~6 frames see the padding differently (3x3 kernels), and the backward GRU
    # carries that tiny edge difference a little way back, so compare away from the edge.
    torch.testing.assert_close(b_batch[1, :45], b_alone[0, :45], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(s_batch[1, :45], s_alone[0, :45], atol=1e-4, rtol=1e-3)


def test_run_model_probabilities():
    m = _model()
    bp, cp = run_model(m, np.random.randn(80, 64).astype(np.float32))
    assert bp.shape == (80,) and cp.shape == (80, N_CLASSES)
    assert ((bp >= 0) & (bp <= 1)).all()
    np.testing.assert_allclose(cp.sum(1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = _model()
    stats = FeatureStats(np.zeros(64, np.float32), np.ones(64, np.float32))
    path = tmp_path / "m.pt"
    save_checkpoint(path, m, stats, {"decode": {"threshold": 0.4, "min_distance_sec": 3.0}})
    m2, stats2, ckpt = load_checkpoint(path)
    x = np.random.randn(50, 64).astype(np.float32)
    np.testing.assert_allclose(run_model(m, x)[0], run_model(m2, x)[0], atol=1e-6)
    assert ckpt["decode"]["threshold"] == 0.4
    np.testing.assert_array_equal(stats2.std, stats.std)


def test_cnn_only_ablation():
    m = StructureNet(n_bands=64, conv_channels=(8, 8, 8), gru_hidden=16, gru_layers=0).eval()
    b, s = m(torch.randn(2, 40, 64), torch.tensor([40, 30]))
    assert b.shape == (2, 40) and s.shape == (2, 40, N_CLASSES)


def test_reverse_padded():
    from sharp.model import reverse_padded
    h = torch.arange(10.0).reshape(2, 5, 1)
    out = reverse_padded(h, torch.tensor([5, 3]))[..., 0]
    assert out.tolist() == [[4, 3, 2, 1, 0], [7, 6, 5, 8, 9]]


def test_bigru_matches_torch_packed_gru():
    """Our BiGRU must equal torch's bidirectional GRU on packed sequences, given the same weights."""
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from sharp.model import BiGRU
    torch.manual_seed(0)
    ours = BiGRU(6, 5, num_layers=2, dropout=0.0).eval()
    ref = torch.nn.GRU(6, 5, num_layers=2, batch_first=True, bidirectional=True).eval()
    with torch.no_grad():
        for layer in range(2):
            for direction, gru in (("", ours.fwd[layer]), ("_reverse", ours.bwd[layer])):
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(ref, f"{name}_l{layer}{direction}").copy_(getattr(gru, f"{name}_l0"))
    x = torch.randn(3, 12, 6)
    lengths = torch.tensor([12, 7, 3])
    with torch.no_grad():
        a = ours(x, lengths)
        packed, _ = ref(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        b, _ = pad_packed_sequence(packed, batch_first=True, total_length=12)
    for i, n in enumerate(lengths.tolist()):
        torch.testing.assert_close(a[i, :n], b[i, :n], atol=1e-6, rtol=1e-5)
