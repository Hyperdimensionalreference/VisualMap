# Windows x64 runtime; models stay installed instead of extracting on every launch.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / 'src'))
RESOURCES = ROOT / 'work/windows/resources'
assets = ROOT / 'src/visualmap/visualization'
datas = [(str(RESOURCES / 'configs'), 'configs'), (str(RESOURCES / 'models'), 'models')]
datas += collect_data_files('visualmap.visualization', includes=[
    '*.html', '*.css', '*.js', 'fonts/*.woff2', 'fonts/OFL.txt'])
hidden = collect_submodules('visualmap') + collect_submodules('tokenizers')
hidden += ['onnxruntime', 'onnxruntime.capi._pybind_state',
           'onnxruntime.capi.onnxruntime_pybind11_state']
binaries = collect_dynamic_libs('onnxruntime')

a = Analysis(
    [str(ROOT / 'src/visualmap/__main__.py')],
    pathex=[str(ROOT / 'src')], binaries=binaries, datas=datas,
    hiddenimports=hidden, hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['tkinter', 'open3d', 'torch', 'torchvision', 'transformers', 'scipy',
              'matplotlib', 'pandas', 'plotly', 'dash', 'pytest', 'IPython',
              'onnxruntime.quantization', 'onnxruntime.transformers',
              'onnxruntime.training', 'huggingface_hub'],
    noarchive=False, optimize=1,
)
# Windows 10/11 always load their system UCRT, even if a local copy exists.
# Keep app-specific VC runtime DLLs; avoid redistributing an unused OS DLL.
a.binaries = [entry for entry in a.binaries if entry[0].lower() != 'ucrtbase.dll']
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='VisualMap',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False,
          version=str(ROOT / 'packaging/version_info.txt'),
          icon=str(ROOT / 'work/windows/VisualMap.ico'))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='VisualMap')
