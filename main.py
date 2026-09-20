import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate

N = "\033[0m"
B = "\033[1m"
DIM = "\033[90m"
CYAN = "\033[36m"
OK = "\033[48;5;40;38;5;16m"
BAD = "\033[48;5;160;38;5;255m"
HEAD = "\033[48;5;236;38;5;255m"


def num(value):
    return f"{int(value):,}"


def cell(style, text, width):
    return style + f"{text:^{width}}" + N


def rule(left, mid, right, widths):
    return left + mid.join("─" * w for w in widths) + right


def print_header(total):
    title = "Lecture 10  ·  Morphological Opening"
    sub = f"{total} images  ·  coin = Positive  ·  sand = Negative"
    width = max(len(title), len(sub)) + 4
    print()
    print(CYAN + "┌" + "─" * width + "┐" + N)
    print(CYAN + "│" + N + B + f"  {title:<{width - 2}}" + N + CYAN + "│" + N)
    print(CYAN + "│" + N + DIM + f"  {sub:<{width - 2}}" + N + CYAN + "│" + N)
    print(CYAN + "└" + "─" * width + "┘" + N)
    print()


def print_matrix(tp, fn, fp, tn):
    w = [16, 22, 22]
    print(B + "  Confusion Matrix" + N + DIM + "  (นับทีละพิกเซล)" + N)
    print("  " + rule("┌", "┬", "┐", w))
    print(
        "  │"
        + cell(HEAD, "", w[0])
        + "│"
        + cell(HEAD, "Predicted +", w[1])
        + "│"
        + cell(HEAD, "Predicted −", w[2])
        + "│"
    )
    print("  " + rule("├", "┼", "┤", w))
    print(
        "  │"
        + cell(HEAD, "Actual + (P)", w[0])
        + "│"
        + cell(OK, f"TP  {num(tp)}", w[1])
        + "│"
        + cell(BAD, f"FN  {num(fn)}", w[2])
        + "│"
    )
    print("  " + rule("├", "┼", "┤", w))
    print(
        "  │"
        + cell(HEAD, "Actual − (N)", w[0])
        + "│"
        + cell(BAD, f"FP  {num(fp)}", w[1])
        + "│"
        + cell(OK, f"TN  {num(tn)}", w[2])
        + "│"
    )
    print("  " + rule("└", "┴", "┘", w))
    print()
    print(
        "  "
        + OK
        + " TP/TN ถูก "
        + N
        + "   "
        + BAD
        + " FN/FP ผิด "
        + N
        + DIM
        + "   + = เหรียญ   − = ทราย"
        + N
    )
    print()


def print_metrics(with_m, without_m):
    tp, fn, fp, tn = with_m["tp"], with_m["fn"], with_m["fp"], with_m["tn"]
    p = tp + fn
    n = fp + tn
    total = p + n
    rows = [
        ("Accuracy ", "(TP+TN) / (P+N)", f"({num(tp)} + {num(tn)}) / {num(total)}", with_m["accuracy"]),
        ("Precision", "TP / (TP+FP)", f"{num(tp)} / ({num(tp)} + {num(fp)})", with_m["precision"]),
        ("Recall   ", "TP / P", f"{num(tp)} / {num(p)}", with_m["recall"]),
        ("F1       ", "2PR / (P+R)", "2 × Precision × Recall / (P+R)", with_m["f1"]),
    ]
    print(B + "  Metrics  ·  Opening vs ground truth" + N)
    print("  " + rule("┌", "┬", "┐", [12, 20, 36, 10]))
    print(
        "  │"
        + cell(HEAD, "metric", 12)
        + "│"
        + cell(HEAD, "สูตร", 20)
        + "│"
        + cell(HEAD, "แทนค่า", 36)
        + "│"
        + cell(HEAD, "ค่า", 10)
        + "│"
    )
    print("  " + rule("├", "┼", "┤", [12, 20, 36, 10]))
    for name, formula, plugged, value in rows:
        print(
            f"  │ {name:<10} │ {formula:<18} │ {plugged:<34} │ {value:>8.4f} │"
        )
    print("  " + rule("└", "┴", "┘", [12, 20, 36, 10]))
    print()
    print(f"  P = {num(p)}   N = {num(n)}   total = {num(total)}")
    print()
    print_bars(with_m["accuracy"], without_m["accuracy"])


def bar(value, width=28):
    filled = max(0, min(width, round(value * width)))
    return CYAN + "█" * filled + N + DIM + "░" * (width - filled) + N


def print_bars(with_acc, without_acc):
    print(B + "  Accuracy comparison" + N)
    print(f"  with Opening     {bar(with_acc)}  {with_acc:.4f}")
    print(f"  without Opening  {bar(without_acc)}  {without_acc:.4f}")
    print()


def draw_confusion(ax, tp, fn, fp, tn):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Predicted +", "Predicted −"])
    ax.set_yticks([1.5, 0.5], ["Actual +", "Actual −"])
    ax.tick_params(length=0, labelsize=11)
    ax.set_title("Confusion Matrix  (pixel count)", pad=12)
    cells = [
        (0, 1, tp, "#c8f0d0", "#146c2e", "TP"),
        (1, 1, fn, "#ffd0d0", "#9b1c1c", "FN"),
        (0, 0, fp, "#ffd0d0", "#9b1c1c", "FP"),
        (1, 0, tn, "#c8f0d0", "#146c2e", "TN"),
    ]
    for x, y, value, face, ink, label in cells:
        ax.add_patch(
            plt.Rectangle((x, y), 1, 1, facecolor=face, edgecolor="white", linewidth=4)
        )
        ax.text(
            x + 0.5,
            y + 0.62,
            label,
            ha="center",
            va="center",
            fontsize=13,
            color=ink,
            fontweight="bold",
        )
        ax.text(
            x + 0.5,
            y + 0.34,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=16,
            color=ink,
            fontweight="bold",
        )
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_roc(ax, fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr)
    fpr_s, tpr_s = fpr[order], tpr[order]
    auc = float(np.trapezoid(tpr_s, fpr_s))
    ax.plot([0, 1], [0, 1], color="#9aa3ad", linestyle="--", linewidth=1.6, label="random")
    ax.plot(fpr_s, tpr_s, color="#0aa2c0", linewidth=2.4, marker="o", markersize=4.5, label="Opening")
    ax.fill_between(fpr_s, tpr_s, alpha=0.12, color="#0aa2c0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("FPR  (False Positive Rate)")
    ax.set_ylabel("TPR  (True Positive Rate)")
    ax.set_title(f"ROC Curve   AUC = {auc:.4f}", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    return auc


def show_plots(result):
    m = result["with_morph"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(12.2, 5.6))
    fig.suptitle(
        f"Lecture 10  ·  Morphological Opening   ({result['total']} images)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    draw_confusion(ax_cm, m["tp"], m["fn"], m["fp"], m["tn"])
    auc = draw_roc(ax_roc, result["fpr"], result["tpr"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).resolve().parent / "result_plot.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(B + "  Graph window" + N + DIM + f"  saved {out.name}   AUC = {auc:.4f}" + N)
    print(DIM + "  กำลังเปิดหน้าต่าง Matplotlib ... ปิดหน้าต่างเมื่อดูเสร็จ" + N)
    backend = plt.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "template"):
        plt.close(fig)
        print(DIM + f"  backend={backend} ไม่เปิดหน้าต่าง — เปิดไฟล์ {out.name} แทน" + N)
    else:
        plt.show()
    print()


def print_result(result):
    m = result["with_morph"]
    print_header(result["total"])
    print_matrix(m["tp"], m["fn"], m["fp"], m["tn"])
    print_metrics(m, result["without_morph"])
    show_plots(result)


def enable_terminal():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


if __name__ == "__main__":
    enable_terminal()
    print(DIM + "evaluating 50 coin images ..." + N)
    print_result(evaluate())
