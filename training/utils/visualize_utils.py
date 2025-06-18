
# print(label_to_class)
def draw_scatter_with_int_legend(X, y, title, label_name_map=None):
    """
    绘制带自定义标签图例的散点图，并将图例在下方每5个一行横排展示。
    Args:
        X: 2D numpy array, shape (n_samples, 2)
        y: 1D numpy array, 整数类别
        title: plot标题
        label_name_map: 可选，{int: str} dict，若给定用作legend名称
    Returns:
        fig: matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    unique_labels = np.unique(y)
    handles = []
    for i, val in enumerate(unique_labels):
        # print(val)
        label_str = str(val)
        # print(label_name_map)
        if label_name_map is not None:
            label_str = label_name_map[int(val)]
        # print(label_str)
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.get_cmap('tab10')(i % 10),
                       markersize=7, label=label_str)
        )
    ncol = 5
    ax.legend(
        handles=handles,
        # title="Labels",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=ncol,
        frameon=True
    )
    # ax.set_title(title)
    plt.tight_layout()
    return fig