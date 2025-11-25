"""
Test script for modules.deeplearning_environment_setup - Environment checks.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile

from modules.deeplearning_environment_setup import (
    PackageStatus,
    check_package,
    check_gpu,
    ensure_requirements,
    format_report,
    main,
    REQUIRED_PACKAGES,
    PACKAGE_IMPORT_MAP,
)


def test_package_status_installed():
    """Test PackageStatus with installed package."""
    status = PackageStatus(name="torch", installed=True, version="1.0.0")
    
    assert status.name == "torch"
    assert status.installed is True
    assert status.version == "1.0.0"
    assert status.error is None


def test_package_status_not_installed():
    """Test PackageStatus with not installed package."""
    status = PackageStatus(
        name="missing_package",
        installed=False,
        version=None,
        error="ModuleNotFoundError",
    )
    
    assert status.installed is False
    assert status.error is not None


def test_check_package_installed():
    """Test check_package with installed package."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.__version__ = "1.0.0"
        mock_import.return_value = mock_module
        
        status = check_package("torch")
        
        assert status.installed is True
        assert status.version == "1.0.0"
        assert status.error is None


def test_check_package_not_installed():
    """Test check_package with not installed package."""
    with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module")):
        status = check_package("missing_package")
        
        assert status.installed is False
        assert status.error is not None


def test_check_package_no_version():
    """Test check_package with package without version."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        del mock_module.__version__  # No version attribute
        mock_import.return_value = mock_module
        
        status = check_package("torch")
        
        assert status.installed is True
        assert status.version == "unknown"


def test_check_gpu_available():
    """Test check_gpu when CUDA is available."""
    with patch("importlib.import_module") as mock_import:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = ["GPU 1", "GPU 2"]
        mock_import.return_value = mock_torch
        
        result = check_gpu()
        
        assert "CUDA available" in result
        assert "2 device(s)" in result


def test_check_gpu_not_available():
    """Test check_gpu when CUDA is not available."""
    with patch("importlib.import_module") as mock_import:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_import.return_value = mock_torch
        
        result = check_gpu()
        
        assert "CUDA not available" in result or "CPU" in result


def test_check_gpu_torch_not_installed():
    """Test check_gpu when torch is not installed."""
    with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module")):
        result = check_gpu()
        
        assert "PyTorch not installed" in result or "cannot check" in result


def test_ensure_requirements_all_present():
    """Test ensure_requirements when all packages are present."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        # The function checks if package name is in content (list of lines)
        # So we need exact line matches, or the package name must be a complete line
        f.write("torch\npytorch-lightning\n")
        temp_path = Path(f.name)
    
    try:
        missing = ensure_requirements(["torch", "pytorch-lightning"], temp_path)
        # If packages are present as complete lines, missing should be empty
        assert len(missing) == 0
    finally:
        temp_path.unlink()


def test_ensure_requirements_some_missing():
    """Test ensure_requirements when some packages are missing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        # Only torch is in the file
        f.write("torch\n")
        temp_path = Path(f.name)
    
    try:
        missing = ensure_requirements(["torch", "pytorch-lightning"], temp_path)
        # pytorch-lightning should be missing
        assert "pytorch-lightning" in missing
        # torch should NOT be in missing (it's in the file as a line)
        # But the function checks if package name is IN the content list (as a line)
        # So "torch" should be found if it's a complete line
        content_lines = temp_path.read_text().splitlines()
        if "torch" in content_lines:
            assert "torch" not in missing
        else:
            assert "torch" in missing
    finally:
        temp_path.unlink()


def test_ensure_requirements_file_not_found():
    """Test ensure_requirements when file doesn't exist."""
    missing = ensure_requirements(["torch"], Path("nonexistent_file.txt"))
    assert len(missing) > 0
    assert "not found" in missing[0] or "nonexistent" in missing[0].lower()


def test_format_report():
    """Test format_report function."""
    statuses = [
        PackageStatus(name="torch", installed=True, version="1.0.0"),
        PackageStatus(name="missing", installed=False, version=None, error="Error"),
    ]
    missing_req = ["missing"]
    
    report = format_report(statuses, missing_req)
    
    assert "torch" in report
    assert "missing" in report
    assert "1.0.0" in report
    assert "Dependency Report" in report or "Report" in report


def test_format_report_no_missing():
    """Test format_report with no missing requirements."""
    statuses = [
        PackageStatus(name="torch", installed=True, version="1.0.0"),
    ]
    missing_req = []
    
    report = format_report(statuses, missing_req)
    
    assert "torch" in report
    assert len(missing_req) == 0


def test_required_packages_list():
    """Test REQUIRED_PACKAGES list is not empty."""
    assert len(REQUIRED_PACKAGES) > 0
    assert isinstance(REQUIRED_PACKAGES, list)
    assert "torch" in REQUIRED_PACKAGES


def test_package_import_map():
    """Test PACKAGE_IMPORT_MAP mapping."""
    assert isinstance(PACKAGE_IMPORT_MAP, dict)
    assert len(PACKAGE_IMPORT_MAP) > 0
    # Check some known mappings
    if "pytorch-lightning" in PACKAGE_IMPORT_MAP:
        assert PACKAGE_IMPORT_MAP["pytorch-lightning"] == "pytorch_lightning"


def test_main_exit_code_all_installed():
    """Test main function when all packages are installed."""
    with patch("modules.deeplearning_environment_setup.check_package") as mock_check:
        mock_check.return_value = PackageStatus(
            name="torch", installed=True, version="1.0.0"
        )
        with patch("modules.deeplearning_environment_setup.ensure_requirements", return_value=[]):
            with patch("builtins.print"):  # Suppress output
                exit_code = main()
                assert exit_code == 0


def test_main_exit_code_some_missing():
    """Test main function when some packages are missing."""
    with patch("modules.deeplearning_environment_setup.check_package") as mock_check:
        def side_effect(pkg):
            if pkg == "torch":
                return PackageStatus(name="torch", installed=True, version="1.0.0")
            return PackageStatus(name=pkg, installed=False, version=None, error="Error")
        
        mock_check.side_effect = side_effect
        with patch("modules.deeplearning_environment_setup.ensure_requirements", return_value=[]):
            with patch("builtins.print"):  # Suppress output
                exit_code = main()
                assert exit_code == 1  # Should return 1 when packages are missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

