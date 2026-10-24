def test_package_imports():
    import gencv
    from gencv.device import get_device
    from gencv.paths import ROOT, load_default_config

    cfg = load_default_config()
    assert cfg["model_id"] == "google/ddpm-cifar10-32"
    assert ROOT.exists()
    assert get_device().type in {"cpu", "mps", "cuda"}
    assert gencv.__version__ == "0.1.0"
