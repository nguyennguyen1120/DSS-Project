"""
Cài dự án ở chế độ editable để mọi `import` trong src/ hoạt động từ bất cứ đâu.

Chạy MỘT LẦN:
    pip install -e .

Sau đó mọi lệnh `python src/generators/rule_based.py ...` đều chạy được
mà không cần `sys.path` hay `PYTHONPATH`.
"""
from setuptools import setup, find_packages

setup(
    name="mcq_dss",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
