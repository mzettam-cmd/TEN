from src.datasets.xor import get_xor

def test_xor_dataset():
    X, y = get_xor()
    assert X.shape == (4, 2)
    assert y.tolist() == [0, 1, 1, 0]
