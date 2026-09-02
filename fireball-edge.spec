# Build on Windows: pyinstaller --clean --noconfirm fireball-edge.spec

from PyInstaller.utils.hooks import collect_dynamic_libs

ort_binaries = collect_dynamic_libs("onnxruntime")

analysis = Analysis(
    ["fireball_edge_entry.py"],
    pathex=["src"],
    binaries=ort_binaries,
    datas=[
        ("edge-config.example.json", "."),
        ("ufocapture-action.example.cmd", "."),
    ],
    hiddenimports=["cv2", "numpy", "onnxruntime"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchvision"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="fireball-edge",
    console=True,
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="fireball-edge",
)
