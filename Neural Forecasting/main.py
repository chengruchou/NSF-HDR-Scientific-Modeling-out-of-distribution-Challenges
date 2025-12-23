import os
import time
import numpy as np
import matplotlib.pyplot as plt


def print_pretty(text):
    print("-------------------")
    print("#---", text)
    print("-------------------")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    candidate = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_stats(path):
    stats = np.load(path)
    if "average" in stats and "std" in stats:
        return stats["average"], stats["std"]
    if "arr_0" in stats:
        data = stats["arr_0"]
    elif len(stats.files) == 1:
        data = stats[stats.files[0]]
    else:
        raise ValueError(f"Unexpected keys in stats file: {stats.files}")
    if data.ndim == 4:
        avg = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True) + 1e-6
        avg = avg.reshape(1, data.shape[2] * data.shape[3])
        std = std.reshape(1, data.shape[2] * data.shape[3])
    else:
        flat = data.reshape(data.shape[0], -1)
        avg = np.mean(flat, axis=0, keepdims=True)
        std = np.std(flat, axis=0, keepdims=True) + 1e-6
    return avg, std


def build_test_input(average, std, num_channels, num_features=9, seed=0):
    flat_avg = average.reshape(1, num_channels, num_features)
    flat_std = std.reshape(1, num_channels, num_features)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, size=flat_avg.shape).astype(np.float32)
    base = flat_avg.astype(np.float32) + noise * flat_std.astype(np.float32)
    x = np.repeat(base[:, None, :, :], 20, axis=1)
    return x


def plot_output(y, monkey_name, channel_idx=0, sample_idx=0, out_dir=None):
    series = y[sample_idx, :, channel_idx]
    plt.figure()
    plt.plot(series)
    plt.title(f"{monkey_name} output (sample {sample_idx}, channel {channel_idx})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    if out_dir:
        out_path = os.path.join(out_dir, f"{monkey_name}_output.png")
        plt.savefig(out_path, dpi=150)
        print("saved plot:", out_path)
    else:
        plt.show()


def run_single(monkey_name, stats_file, weight_file, out_dir=None):
    from model import Model

    stats_path = resolve_path(stats_file)
    avg, std = load_stats(stats_path)

    model = Model(monkey_name)
    model.weight_file = weight_file
    model.load()

    x = build_test_input(avg, std, model.input_size)
    y = model.predict(x)

    print_pretty(f"{monkey_name} output")
    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("output stats:", float(np.min(y)), float(np.max(y)), float(np.mean(y)))
    first = y[:, :10, :]
    last = y[:, 10:, :]
    print("first10 stats:", float(np.min(first)), float(np.max(first)), float(np.mean(first)), float(np.std(first)))
    print("last10 stats:", float(np.min(last)), float(np.max(last)), float(np.mean(last)), float(np.std(last)))
    plot_output(y, monkey_name, channel_idx=0, sample_idx=0, out_dir=out_dir)


def main():
    start = time.time()

    out_dir = os.path.dirname(__file__)
    run_single(
        "affi",
        r"Neural Forecasting\dataset\train_data_affi.npz",
        "model_affi_v2.pth",
        out_dir=out_dir,
    )
    run_single(
        "beignet",
        r"Neural Forecasting\dataset\train_data_beignet.npz",
        "model_beignet_v2.pth",
        out_dir=out_dir,
    )

    duration = time.time() - start
    print_pretty(f"Total duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
