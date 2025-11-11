"""Tests for configuration management."""

import pytest
import tempfile
import os
from deepsdf.utils.config import Config, ModelConfig, TrainingConfig, DataConfig


def test_default_config() -> None:
    """Test default configuration."""
    config = Config()
    
    assert config.model.latent_size == 256
    assert config.training.batch_size == 32
    assert config.data.data_dir == "./data"


def test_config_from_dict() -> None:
    """Test creating config from dictionary."""
    config_dict = {
        'model': {'latent_size': 128},
        'training': {'batch_size': 16},
        'experiment_name': 'test_experiment',
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.model.latent_size == 128
    assert config.training.batch_size == 16
    assert config.experiment_name == 'test_experiment'


def test_config_to_dict() -> None:
    """Test converting config to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert 'model' in config_dict
    assert 'training' in config_dict
    assert 'data' in config_dict
    assert config_dict['model']['latent_size'] == 256


def test_config_yaml_io() -> None:
    """Test saving and loading config from YAML."""
    config = Config()
    config.experiment_name = 'yaml_test'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.to_yaml(f.name)
        yaml_path = f.name
    
    try:
        loaded_config = Config.from_yaml(yaml_path)
        assert loaded_config.experiment_name == 'yaml_test'
        assert loaded_config.model.latent_size == config.model.latent_size
    finally:
        os.unlink(yaml_path)


def test_config_json_io() -> None:
    """Test saving and loading config from JSON."""
    config = Config()
    config.experiment_name = 'json_test'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.to_json(f.name)
        json_path = f.name
    
    try:
        loaded_config = Config.from_json(json_path)
        assert loaded_config.experiment_name == 'json_test'
        assert loaded_config.model.latent_size == config.model.latent_size
    finally:
        os.unlink(json_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
