
import ast
from typing import Literal
from collections.abc import Sequence

import colorsys
import matplotlib.pyplot as plt
import polars as pl
import numpy as np


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        return np.array(ast.literal_eval(series))


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


def load_xs_ys_avg_y(
        file: str,
        model_scale: float | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        qk_activ: bool | None = None, 
        qk_norm: bool | None = None, 
        attn_activ: bool | None = None, 
        post_attn_norm: bool | None = None,
        num_heads: int | None = None,
        run_num: int | None = None,
        seed: int | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs", "train_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("last_val_loss").ge(0))  # initial condition -> always true

    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if qk_activ is not None:
        filters &= (pl.col("qk_activ") == qk_activ)
    if qk_norm is not None:
        filters &= (pl.col("qk_norm") == qk_norm)
    if attn_activ is not None:
        filters &= (pl.col("attn_activ") == attn_activ)
    if post_attn_norm is not None:
        filters &= (pl.col("post_attn_norm") == post_attn_norm)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if run_num is not None:
        filters &= (pl.col("run_num") == run_num)
    if seed is not None:
        filters &= (pl.col("seed") == seed)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])
    xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    avg_ys = np.mean(ys, axis=0)
    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["epoch"][i]) for i in range(len(df["epoch"]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["tokens_seen"][i]) for i in range(len(df["tokens_seen"]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["cumulative_time"][i]) for i in range(len(df["cumulative_time"]))]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
        .select(*[pl.col(target) for target in targets])
        .collect()
        .unique()
    )
    # Sort combinations alphabetically by content, target by target (for consistency in plotting)
    for target in targets:
        combinations = combinations.sort(target)
    # Create a list of settings
    for features in zip(
            *[combinations[target] for target in targets]
    ):
        settings.append(tuple(features))

    return settings


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def unique_num_params(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )

QK_ACTIV_TYPE = Literal["gelu", "none"]
QK_NORM_TYPE = Literal["fro_norm", "rms_norm", "none"]
ATTN_ACTIV_TYPE = Literal["softmax", "sigmoid", "tanh", "none"]
POST_ATTN_NORM_TYPE = Literal["rms_norm", "none"]


def plot_results(
        file: str,
        depth: int | Sequence[int] | None = 8,
        width: int | Sequence[int] | None = 384,
        num_heads: int | Sequence[int] | None = None,
        qk_activ: QK_ACTIV_TYPE | Sequence[QK_ACTIV_TYPE] | None = None,
        qk_norm: QK_NORM_TYPE | Sequence[QK_NORM_TYPE] | None = None,
        attn_activ: ATTN_ACTIV_TYPE | Sequence[ATTN_ACTIV_TYPE] | None = None,
        post_attn_norm: POST_ATTN_NORM_TYPE | Sequence[POST_ATTN_NORM_TYPE] | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
) -> None:
    settings = get_unique_settings(file, ["num_heads", "qk_activ", "qk_norm", "attn_activ", "post_attn_norm", "depth", "width"])

    depth = [depth] if isinstance(depth, int) else depth
    width = [width] if isinstance(width, int) else width
    num_heads = [num_heads] if isinstance(num_heads, int) else num_heads
    qk_activ = [qk_activ] if isinstance(qk_activ, str) else qk_activ
    qk_norm = [qk_norm] if isinstance(qk_norm, str) else qk_norm
    attn_activ = [attn_activ] if isinstance(attn_activ, str) else attn_activ
    post_attn_norm = [post_attn_norm] if isinstance(post_attn_norm, str) else post_attn_norm

    if num_heads is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if nh in num_heads]
    if qk_activ is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if qka in qk_activ]
    if qk_norm is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if qkn in qk_norm]
    if attn_activ is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if aa in attn_activ]
    if post_attn_norm is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if pan in post_attn_norm]
    if depth is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if d in depth]
    if width is not None:
        settings = [(nh, qka, qkn, aa, pan, d, w) for nh, qka, qkn, aa, pan, d, w in settings if w in width]

    colors = generate_distinct_colors(len(settings))

    for color, (num_heads_, qk_activ_, qk_norm_, attn_activ_, post_attn_norm_, depth_, width_) in zip(colors, settings):
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                depth=depth_,
                width=width_,
                num_heads=num_heads_,
                qk_activ=qk_activ_,
                qk_norm=qk_norm_,
                attn_activ=attn_activ_,
                post_attn_norm=post_attn_norm_,
                to_plot=to_plot,
                plot_over=plot_over,
            )
            if plot_all:
                for y in ys:
                    if loglog:
                        plt.loglog(xs, y, color=color, alpha=0.2)
                    else:
                        plt.plot(xs, y, color=color, alpha=0.2)

            num_params = pl.scan_csv(file).filter(
                (pl.col("num_heads") == num_heads_)
                & (pl.col("qk_activ") == qk_activ_)
                & (pl.col("qk_norm") == qk_norm_)
                & (pl.col("attn_activ") == attn_activ_)
                & (pl.col("post_attn_norm") == post_attn_norm_)
                & (pl.col("depth") == depth_)
                & (pl.col("width") == width_)
            ).collect()["num_params"][0]
            
            label = (
                f"qk_activ={qk_activ_}, qk_norm={qk_norm_}, attn_activ={attn_activ_}, post_attn_norm={post_attn_norm_}, "
                f"h={num_heads_}, d={depth_}, w={width_}, p={format_num_params(num_params)}"
            )
            if loglog:
                plt.loglog(xs, avg_ys, color=color if plot_all else None, label=label)
            else:
                plt.plot(xs, avg_ys, color=color if plot_all else None, label=label)


    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        # You should probably adjust the filename
        plt.savefig(f"{to_plot}_vs_{plot_over}.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


if __name__ == "__main__":
    file = "results/results1.csv"

    plot_results(
        file=file,
        depth=None,
        width=None,
        num_heads=None,
        qk_activ=["gelu", "none"],
        qk_norm=["rms_norm", "none"],
        attn_activ=["sigmoid", "softmax"],
        post_attn_norm="none",
        to_plot="val_loss",
        plot_over="epoch",
        show=True,
        loglog=False,
        plot_all=False,
    )
