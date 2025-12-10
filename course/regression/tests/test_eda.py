import pytest
import pandas as pd
from plotly.graph_objects import Figure
from course.regression.eda import (
    _boxplot,
    boxplot_age,
    boxplot_rooms,
    shortfall_distribution,
    scatter_rooms_shortfall,
    shortfall_by_authority,
    mean_shortfall_by_authority,
    mean_shortfall_age,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [10, 15, 20, 25]
    })


def test_boxplot_returns_figure(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert isinstance(fig, Figure)


def test_boxplot_title(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert fig.layout.title.text == 'Test Title'


def test_boxplot_axes(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert fig.data[0].x is not None
    assert fig.data[0].y is not None


def test_boxplot_age_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    boxplot_age()
    assert (tmp_path / "boxplot_age.html").exists()


def test_boxplot_rooms_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    boxplot_rooms()
    assert (tmp_path / "boxplot_rooms.html").exists()


def test_shortfall_distribution_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    shortfall_distribution()
    assert (tmp_path / "shortfall_distribution.html").exists()


def test_scatter_rooms_shortfall_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    scatter_rooms_shortfall()
    assert (tmp_path / "scatter_rooms_shortfall.html").exists()


def test_shortfall_by_authority_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    shortfall_by_authority()
    assert (tmp_path / "shortfall_by_authority.html").exists()


def test_mean_shortfall_by_authority_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    mean_shortfall_by_authority()
    assert (tmp_path / "mean_shortfall_by_authority.html").exists()


def test_mean_shortfall_age_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("course.regression.eda.VIGNETTE_DIR", tmp_path)
    mean_shortfall_age()
    assert (tmp_path / "mean_shortfall_age.html").exists()
