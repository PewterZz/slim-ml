from slim_ml.budget import StaticBudget, Tier


def test_reserve_within_capacity():
    b = StaticBudget({Tier.RAM: 1024, Tier.VRAM: 0, Tier.NVME: 0, Tier.DISK: 0})
    assert b.reserve(Tier.RAM, 512, "a") is True
    assert b.capacity(Tier.RAM).available_bytes == 512
    assert abs(b.pressure(Tier.RAM) - 0.5) < 1e-6


def test_reserve_beyond_capacity_fails():
    b = StaticBudget({Tier.RAM: 1024, Tier.VRAM: 0, Tier.NVME: 0, Tier.DISK: 0})
    assert b.reserve(Tier.RAM, 2048, "a") is False
    assert b.capacity(Tier.RAM).reserved_bytes == 0


def test_release_frees_bytes():
    b = StaticBudget({Tier.RAM: 1024, Tier.VRAM: 0, Tier.NVME: 0, Tier.DISK: 0})
    b.reserve(Tier.RAM, 512, "a")
    b.release(Tier.RAM, 256, "a")
    assert b.capacity(Tier.RAM).available_bytes == 768


def test_headroom_reduces_capacity():
    b = StaticBudget(
        {Tier.RAM: 1024, Tier.VRAM: 0, Tier.NVME: 0, Tier.DISK: 0},
        headroom_bytes={Tier.RAM: 256},
    )
    assert b.capacity(Tier.RAM).total_bytes == 768
