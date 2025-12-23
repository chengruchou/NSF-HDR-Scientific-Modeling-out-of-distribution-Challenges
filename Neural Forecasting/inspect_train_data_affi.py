import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    data = np.load(path)
    if "arr_0" in data:
        return data["arr_0"]
    if len(data.files) == 1:
        return data[data.files[0]]
    raise ValueError(f"Unexpected keys in {path}: {data.files}")


def main():
    base = os.path.dirname(__file__)
    npz_path = os.path.join(base, "dataset", "train_data_affi.npz")
    x = load_npz(npz_path)

    print("value meaning:")
    print("  B: samples, T: timesteps, C: channels, F: features")
    print("  x shape uses (B,T,C,F) when 4D; otherwise raw values")
    print("  feature0 means the first feature (index 0) used for forecasting")
    print("  overall stats -> min/max/mean/std over all values in x")
    print("  feature0 stats -> min/max/mean/std over only feature0 values")

    print("data shape:", x.shape, "dtype:", x.dtype)
    print("overall stats:", float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x)))

    if x.ndim == 4:
        feature0 = x[..., 0]
        print("feature0 stats:", float(np.min(feature0)), float(np.max(feature0)),
              float(np.mean(feature0)), float(np.std(feature0)))
    else:
        feature0 = x

    plt.figure()
    plt.hist(feature0.ravel(), bins=100)
    plt.title("train_data_affi feature0 distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = os.path.join(base, "train_data_affi_feature0_hist.png")
    plt.savefig(out_path, dpi=150)
    print("saved plot:", out_path)


if __name__ == "__main__":
    main()
