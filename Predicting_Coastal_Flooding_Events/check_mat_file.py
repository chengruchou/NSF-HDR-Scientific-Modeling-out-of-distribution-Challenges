import argparse
from pathlib import Path


def try_load_with_scipy(path):
    try:
        import scipy.io as sio
    except Exception as exc:  # pragma: no cover - import error path
        return False, f"scipy.io import failed: {exc}"
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        return False, f"scipy.io.loadmat failed: {exc}"
    return True, data


def try_load_with_h5py(path):
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - import error path
        return False, f"h5py import failed: {exc}"
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
    except Exception as exc:
        return False, f"h5py.File failed: {exc}"
    return True, keys


def summarize_mat(data):
    meta_keys = {"__header__", "__version__", "__globals__"}
    keys = [k for k in data.keys() if k not in meta_keys]
    print(f"Found {len(keys)} variable(s).")
    for key in keys:
        value = data[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"- {key}: type={type(value).__name__}, shape={shape}, dtype={dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a .mat file and print basic variable information."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="NEUSTG_19502020_12stations.mat",
        help="Path to .mat file (default: NEUSTG_19502020_12stations.mat)",
    )
    args = parser.parse_args()
    mat_path = Path(args.path)

    if not mat_path.exists():
        raise SystemExit(f"File not found: {mat_path}")

    ok, result = try_load_with_scipy(str(mat_path))
    if ok:
        print("Loaded with scipy.io.loadmat")
        summarize_mat(result)
        return

    print(result)
    ok, result = try_load_with_h5py(str(mat_path))
    if ok:
        print("Loaded with h5py (likely MATLAB v7.3).")
        print(f"Top-level groups/datasets: {result}")
        return

    raise SystemExit(f"Unable to read .mat file with scipy or h5py: {result}")


if __name__ == "__main__":
    main()
