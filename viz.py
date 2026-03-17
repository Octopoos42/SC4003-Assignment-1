import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from const import color_map, direction_map

def plot_utility_evolution(utility_history):
    """
    Create an interactive Plotly line chart showing utility estimates over iterations.

    Parameters
    ----------
    utility_history : dict
        Mapping from state (tuple[int, int] or other hashable) to a list of numeric
        utility estimates recorded at each iteration. Example: {(0,0): [0, 1.2, 1.5], ...}.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly Figure containing one line per state. The x-axis is iteration index
        and the y-axis is the utility estimate.

    Notes
    -----
    - Requires `plotly.graph_objects` imported as `go`.
    - Each state's series is plotted with `mode="lines"` and labeled using the state key.
    - The figure uses a white template and unified hover mode for easy comparison.
    """

    fig = go.Figure()
    
    for state, values in utility_history.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines",
                name=f"State {state}",
            )
        )
    
    fig.update_layout(
        xaxis_title="Iterations",
        yaxis_title="Utility Estimates",
        legend=dict(x=1.05, y=1.0),
        template="plotly_white",
        hovermode="x unified",
    )
    
    return fig

def plot_maze(maze, valid_states, utility_history, policy_history, iteration, title):
    """
    Draw the maze grid with colored cells, utility values, and policy arrows for a given iteration.

    Parameters
    ----------
    maze : np.ndarray
        2D array representing the maze layout. Values in the array are used as keys
        into `color_map` and `rewards` elsewhere.
    valid_states : iterable[tuple[int, int]]
        Iterable of (row, col) coordinates that should be drawn (non-wall cells).
    utility_history : dict
        Mapping from state to list of utility estimates per iteration. Used to
        display the utility value for `iteration` if available.
    policy_history : list
        List of policy snapshots (each a dict mapping state to action). The function
        reads `policy_history[iteration]` to draw action arrows.
    iteration : int
        Index of the iteration whose utilities and policy should be displayed.
    title : str
        Title string for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object containing the maze visualization.

    Notes
    -----
    - Expects the following globals to be defined in the calling module:
      **color_map** : dict mapping cell type (string) to a Matplotlib color;
      **direction_map** : dict mapping action symbols (e.g., "U","D","L","R") to arrow glyphs or strings.
    - Uses `matplotlib.patches.Rectangle` to draw cells and `ax.text` to annotate utilities
      and policy arrows. Coordinates assume each cell is 1x1 and text is centered.
    - The plot disables axis ticks and flips the y-axis so row 0 appears at the top.
    """

    fig, ax = plt.subplots(figsize=(12, 12))
    
    for r, c in valid_states:
        cell_type = str(maze[r, c])
        ax.add_patch(patches.Rectangle((c, r), 1, 1, color=color_map[cell_type]))
        
        if (r, c) in utility_history and len(utility_history[(r, c)]) > iteration:
            ax.text(
                c + 0.5,
                r + 0.5,
                f"{utility_history[(r, c)][iteration]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
        
        if (r, c) in policy_history[iteration]:
            ax.text(
                c + 0.5,
                r + 0.2,
                direction_map[policy_history[iteration].get((r, c), ".")],
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, maze.shape[1])
    ax.set_ylim(maze.shape[0], 0)
    ax.set_title(title)
    
    return fig