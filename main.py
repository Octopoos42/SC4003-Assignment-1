import streamlit as st
from grid import load_preset_grid, create_or_resize_grid, display_grid_editor
from algo import run_mdp
from viz import plot_utility_evolution, plot_maze

def configure_page():
    """
    Configure the Streamlit page for the MDP solver application.

    Behavior
    --------
    - Sets the browser tab title and layout via `st.set_page_config`.
    - Renders the main page title using `st.title`.

    Notes
    -----
    - Should be called once at app startup to ensure consistent page metadata.
    """

    st.set_page_config(page_title="MDP Solver - SC4003 Assignment", layout="centered")
    st.title("SC4003 Assignment 1 - Interactive Visualization")


def initialize_session_state():
    """
    Initialize required keys in Streamlit session state.

    Behavior
    --------
    - Creates `st.session_state["saved_grids"]` as an empty dict if it does not exist.

    Notes
    -----
    - Use this at app startup to avoid KeyError when accessing saved grids later.
    """

    if "saved_grids" not in st.session_state:
        st.session_state["saved_grids"] = {}


def manage_grid():
    """
    Manage grid UI flow: load preset, create/resize grid, and display editor.

    Returns
    -------
    np.ndarray
        The current grid stored in session state after any user interactions.

    Behavior
    --------
    - Calls `load_preset_grid` to let the user pick and load a preset.
    - Calls `create_or_resize_grid` to allow resizing or creating a new grid.
    - Calls `display_grid_editor` to render the clickable grid editor.
    - Returns the final grid object for downstream processing.

    Notes
    -----
    - Relies on other helper functions to perform the actual UI work.
    """

    load_preset_grid()
    grid = create_or_resize_grid()
    display_grid_editor(grid)
    
    return grid


def configure_rewards():
    """
    Render reward configuration controls and return the rewards mapping.

    Returns
    -------
    dict
        Mapping from tile symbol to numeric reward, e.g., {".": -0.05, "G": 1.0, "B": -1.0, "S": -0.05}.

    Behavior
    --------
    - Displays a subheader and two-column inputs for reward values.
    - Reads numeric inputs for Empty (.), Goal (G), Bad (B), and Start (S).

    Notes
    -----
    - Values are returned immediately and can be passed to MDP routines.
    """

    st.subheader("Reward Configuration")
    rewards = {}
    col1, col2 = st.columns(2)
    
    with col1:
        rewards["."] = st.number_input("Reward for Empty (.)", value=-0.05)
        rewards["G"] = st.number_input("Reward for Goal (G)", value=1.0)
    
    with col2:
        rewards["B"] = st.number_input("Reward for Bad State (B)", value=-1.0)
        rewards["S"] = st.number_input("Reward for Start (S)", value=-0.05)
    
    return rewards


def configure_convergence_threshold():
    """
    Render controls for selecting convergence constant C and an option to run all C values.

    Returns
    -------
    tuple
        (C_selection: float, run_all_c: bool, c_values: list[float])

    Behavior
    --------
    - Shows a selectbox with preset C values and a checkbox to compute convergence
      steps for all listed C values.
    - Returns the chosen C, the boolean flag, and the list of candidate C values.

    Notes
    -----
    - `run_all_c` may trigger additional, potentially slower computations.
    """

    st.subheader("Convergence Threshold (C)")
    c_values = [0.0001, 0.001, 0.01, 0.1]
    C_selection = st.selectbox("Choose one C to run the MDP with:", c_values, index=2)
    run_all_c = st.checkbox("Compute convergence steps for all c-values")

    return C_selection, run_all_c, c_values


def prepare_mdp(grid):
    """
    Prepare MDP state lists from the provided grid.

    Parameters
    ----------
    grid : np.ndarray
        2D array representing the current grid layout stored in session state.

    Returns
    -------
    maze : np.ndarray
        The grid used as the maze representation (same as input or session copy).
    valid_states : list[tuple[int, int]]
        All (row, col) coordinates in the grid.
    valid_non_wall_states : list[tuple[int, int]]
        Subset of valid_states excluding wall cells ("W").

    Behavior
    --------
    - Reads the grid from `st.session_state` if present, otherwise uses the provided `grid`.
    - Constructs lists of coordinates for downstream MDP algorithms.
    """

    maze = st.session_state.get("grid", grid)
    rows, cols = maze.shape
    valid_states = [(r, c) for r in range(rows) for c in range(cols)]
    valid_non_wall_states = [(r, c) for (r, c) in valid_states if maze[r, c] != "W"]

    return maze, valid_states, valid_non_wall_states


def run_mdp_on_request(grid, maze, valid_non_wall_states, rewards, C_selection):
    """
    Run the MDP algorithms when the user requests it and save results to session state.

    Parameters
    ----------
    grid : np.ndarray
        Current grid layout (used to persist any last-minute changes).
    maze : np.ndarray
        Maze representation passed to the MDP solver.
    valid_non_wall_states : list[tuple[int, int]]
        States that are not walls and should be considered by the MDP.
    rewards : dict
        Reward mapping for tile types.
    C_selection : float
        Convergence constant to pass to the MDP solver.

    Behavior
    --------
    - When the "Run MDP" button is pressed:
      * Persists the current grid to `st.session_state["grid"]`.
      * Calls `run_mdp` to compute value iteration and policy iteration results.
      * Stores the returned results in `st.session_state["mdp_results"]`.
      * Displays a success message.

    Notes
    -----
    - `run_mdp` is expected to return a tuple containing utilities, policies,
      histories, and iteration counts for both algorithms.
    """

    if st.button("Run MDP"):
        st.session_state["grid"] = grid
        mdp_results = run_mdp(rewards, maze, valid_non_wall_states, gamma=0.99, max_iters=2000, C=C_selection)
        st.session_state["mdp_results"] = mdp_results
        st.success("MDP computation completed!")


def compute_convergence_for_all_c(run_all_c, c_values, rewards, maze, valid_non_wall_states):
    """
    Compute and display convergence steps for each C value when requested.

    Parameters
    ----------
    run_all_c : bool
        If True, compute convergence for every value in `c_values`.
    c_values : list[float]
        List of convergence constants to evaluate.
    rewards : dict
        Reward mapping for tile types.
    maze : np.ndarray
        Maze representation.
    valid_non_wall_states : list[tuple[int, int]]
        States considered by the MDP.

    Behavior
    --------
    - Iterates over `c_values` and calls `run_mdp` for each C.
    - Collects the number of iterations required for convergence for both
      value iteration and policy iteration.
    - Displays a results table in the Streamlit app.

    Notes
    -----
    - This can be computationally expensive depending on grid size and number of C values.
    """

    if run_all_c:
        st.markdown("## Convergence Steps for Each C Value")
        results_table = []
        
        for c_val in c_values:
            (uvi, pvi, uhvi, phvi, steps_vi, upi, ppi, uhpi, phpi, steps_pi) = run_mdp(
                rewards, maze, valid_non_wall_states, gamma=0.99, max_iters=2000, C=c_val
            )
            results_table.append({
                "C": c_val,
                "Value Iteration Convergence": steps_vi,
                "Policy Iteration Convergence": steps_pi,
            })
        
        st.write("Results Table (Number of Iterations):")
        st.table(results_table)


def display_mdp_results():
    """
    Fetch MDP results from session state and return them for visualization.

    Returns
    -------
    tuple or None
        If results exist, returns:
        (utilities_vi, policy_vi, utility_history_vi, policy_history_vi, max_iters_vi,
         utilities_pi, policy_pi, utility_history_pi, policy_history_pi, max_iters_pi)
        Otherwise returns None after showing an informational message.

    Behavior
    --------
    - Checks `st.session_state["mdp_results"]` and unpacks the stored tuple.
    - If no results are present, displays an info message prompting the user to run the MDP.

    Notes
    -----
    - Designed to be called before visualization routines that expect these values.
    """

    if "mdp_results" in st.session_state and st.session_state["mdp_results"] is not None:
        (
            utilities_vi,
            policy_vi,
            utility_history_vi,
            policy_history_vi,
            max_iters_vi,
            utilities_pi,
            policy_pi,
            utility_history_pi,
            policy_history_pi,
            max_iters_pi,
        ) = st.session_state["mdp_results"]
        
        return utilities_vi, policy_vi, utility_history_vi, policy_history_vi, max_iters_vi, utilities_pi, policy_pi, utility_history_pi, policy_history_pi, max_iters_pi
    
    else:
        st.info("Please run the MDP first to see the results.")


def visualize_results(maze, valid_states, utility_history_vi, policy_history_vi, max_iters_vi, utility_history_pi, policy_history_pi, max_iters_pi):
    """
    Visualize MDP outputs for policy iteration and value iteration.

    Parameters
    ----------
    maze : np.ndarray
        Maze layout used for plotting.
    valid_states : list[tuple[int, int]]
        Coordinates of drawable cells.
    utility_history_vi : dict
        Utility history per state for value iteration.
    policy_history_vi : list
        Policy snapshots per iteration for value iteration.
    max_iters_vi : int
        Number of iterations performed for value iteration.
    utility_history_pi : dict
        Utility history per state for policy iteration.
    policy_history_pi : list
        Policy snapshots per iteration for policy iteration.
    max_iters_pi : int
        Number of iterations performed for policy iteration.

    Behavior
    --------
    - Confirms MDP results exist in session state.
    - Creates two tabs: "Policy Iteration" and "Value Iteration".
    - For each tab:
      * Shows a Plotly chart of utility evolution using `plot_utility_evolution`.
      * Provides a slider to select an iteration and displays a Matplotlib maze
        snapshot using `plot_maze`.

    Notes
    -----
    - Expects `plot_utility_evolution` and `plot_maze` to be available in the module.
    - Uses Streamlit tabs, sliders, and plotting helpers to create an interactive UX.
    """

    if "mdp_results" in st.session_state and st.session_state["mdp_results"] is not None:
        tabs = st.tabs(["Policy Iteration", "Value Iteration"])

        with tabs[0]:
            st.markdown("### Utility Evolution Over Iterations (Policy Iteration)")
            st.plotly_chart(plot_utility_evolution(utility_history_pi), width="stretch")
            st.markdown("### Select Iteration to View Policy and Utility Grid")
            iteration_pi = st.slider("Select Iteration (Policy Iteration)", 0, max_iters_pi - 1, max_iters_pi - 1)
            st.pyplot(plot_maze(maze, valid_states, utility_history_pi, policy_history_pi, iteration_pi,
                                f"Policy and Utilities at Iteration {iteration_pi}"))

        with tabs[1]:
            st.markdown("### Utility Evolution Over Iterations (Value Iteration)")
            st.plotly_chart(plot_utility_evolution(utility_history_vi), width="stretch")
            st.markdown("### Select Iteration to View Policy and Utility Grid")
            iteration_vi = st.slider("Select Iteration (Value Iteration)", 0, max_iters_vi - 1, max_iters_vi - 1)
            st.pyplot(plot_maze(maze, valid_states, utility_history_vi, policy_history_vi, iteration_vi,
                                f"Value and Utilities at Iteration {iteration_vi}"))


# -------------------------------
# Page Script Execution
# -------------------------------
configure_page()
initialize_session_state()
grid = manage_grid()
rewards = configure_rewards()
C_selection, run_all_c, c_values = configure_convergence_threshold()
maze, valid_states, valid_non_wall_states = prepare_mdp(grid)
run_mdp_on_request(grid, maze, valid_non_wall_states, rewards, C_selection)
compute_convergence_for_all_c(run_all_c, c_values, rewards, maze, valid_non_wall_states)
mdp_results = display_mdp_results()

if mdp_results is not None:
    utilities_vi, policy_vi, utility_history_vi, policy_history_vi, max_iters_vi, utilities_pi, policy_pi, utility_history_pi, policy_history_pi, max_iters_pi = mdp_results
    visualize_results(maze, valid_states, utility_history_vi, policy_history_vi, max_iters_vi,
                      utility_history_pi, policy_history_pi, max_iters_pi)