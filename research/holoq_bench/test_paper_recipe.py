"""Check the paper contract independently of the legacy reconstruction helper."""
import pytest
import torch
from research.holoq_bench.build_paper_holoq import paper_rotation, paper_zigzag, gptq_signed15


def test_largest_channels_are_distributed_across_blocks():
    energy = torch.arange(256, 0, -1, dtype=torch.float32)
    perm = paper_zigzag(energy)
    assert torch.equal(perm.sort().values, torch.arange(256))
    groups = perm.reshape(4, 64)
    # All four highest-energy channels must belong to different blocks. The
    # legacy global-high/low interleave fails this cross-block requirement.
    assert set(groups[:, 0].tolist()) == {0, 1, 2, 3}
    sums = energy[groups].sum(1)
    assert torch.equal(sums, sums[:1].expand(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")
def test_input_rotation_matches_independent_svd_times_random_signs_and_hadamard():
    torch.manual_seed(21)
    w = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    wr, r, p = paper_rotation(w, 37)
    wp = w.float()[:, p]
    u = torch.linalg.svd(wp.reshape(256, 2, 64).permute(1, 2, 0), full_matrices=False).U
    largest = u.abs().argmax(-2, keepdim=True)
    u = u * u.gather(-2, largest).sign()
    generator = torch.Generator().manual_seed(37)
    signs = (torch.randint(0, 2, (2, 64), generator=generator) * 2 - 1).float().cuda()
    h = torch.tensor([[1.]], device="cuda")
    while len(h) < 64:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    expected = ((u * signs[:, None, :]) @ (h / 8)).half()
    torch.testing.assert_close(r, expected, rtol=.002, atol=.002)
    x = torch.randn(7, 128, device="cuda")
    xr = torch.bmm(x[:, p].reshape(7, 2, 64).transpose(0, 1), r.float()).transpose(0, 1).reshape(7, 128)
    actual = xr @ wr.t()
    reference = x @ w.float().t()
    nmse = ((actual - reference).square().sum() / reference.square().sum()).item()
    assert nmse < 2e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")
def test_official_gptq_solver_uses_literal_paper_signed15_grid():
    torch.manual_seed(22)
    w = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    x = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    q, s = gptq_signed15(w, x)
    assert q.min().item() == -7 and q.max().item() == 7
    assert torch.equal(s, s.half().float())
    assert torch.isfinite((q.float() * s).half()).all()
