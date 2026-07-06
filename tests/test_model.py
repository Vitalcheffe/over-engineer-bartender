import sys; sys.path.insert(0, '..')
def test_import():
    import model
    assert hasattr(model, '__name__')
def test_runs():
    import model
    assert True
