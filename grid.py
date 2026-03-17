import streamlit as st
import numpy as np
from streamlit_extras.stylable_container import stylable_container
from const import preset_grids, color_map, tile_options

def load_preset_grid():
    """
    Load a preset grid into the Streamlit session state.

    Behavior
    --------
    - Presents a selectbox for choosing a preset grid from `preset_grids`.
    - When the user clicks "Load Preset", stores the selected grid in
      `st.session_state["grid"]`, updates `rows` and `cols` to match the grid
      shape, clears any previous MDP results (`st.session_state["mdp_results"] = None`),
      shows a success message, and triggers a rerun.

    Notes
    -----
    - Expects `preset_grids` to be a dict mapping preset names to numpy arrays.
    - Relies on Streamlit's session state to persist the selected grid and related
      UI state across reruns.
    """

    st.subheader("Load Preset Grid")
    preset_choice = st.selectbox("Choose a preset grid:", list(preset_grids.keys()))
   
    if st.button("Load Preset"):
        selected_grid = preset_grids[preset_choice]
        st.session_state["grid"] = selected_grid
        st.session_state["rows"] = selected_grid.shape[0]
        st.session_state["cols"] = selected_grid.shape[1]
        st.session_state["mdp_results"] = None
        st.success(f"Preset grid '{preset_choice}' loaded!")
        st.rerun()

def create_or_resize_grid():
    """
    Create or resize the editable grid stored in Streamlit session state.

    Behavior
    --------
    - Renders explanatory text describing tile types and how clicking cycles them.
    - Shows sliders to choose number of rows and columns (3 to 10).
    - If the current session grid is missing or its shape differs from the chosen
      size, constructs a new grid filled with "." and copies over the overlapping
      region from the existing grid or a default preset.
    - Saves the resulting grid to `st.session_state["grid"]`.

    Returns
    -------
    np.ndarray
        The grid currently stored in `st.session_state["grid"]`.

    Notes
    -----
    - Uses `preset_grids["Part 1"]` as a fallback when no existing grid is present.
    - Preserves as many existing cell values as fit in the new dimensions.
    - Does not itself render clickable tiles; that is handled by `display_grid_editor`.
    """

    st.subheader("Edit Grid")
    st.write(
        """
        Each tile represents a different state in the environment. Clicking on a tile **cycles through** the following options:

        - **G (Goal)** - The destination state that gives the agent a positive reward.
        - **. (Empty Space)** - A neutral tile the agent can traverse freely.
        - **B (Bad State)** - A harmful tile that yields a negative reward (e.g., a danger zone).
        - **S (Start State)** - The agent's starting location.
        - **W (Wall)** - An impassable tile that the agent cannot enter.

        Click any tile to change its type and customize the grid layout.
    """
    )
    rows = st.slider("Grid Rows", 3, 20, st.session_state.get("rows", 6))
    cols = st.slider("Grid Columns", 3, 20, st.session_state.get("cols", 6))
   
    if "grid" not in st.session_state or st.session_state["grid"].shape != (rows, cols):
        new_grid = np.full((rows, cols), ".", dtype=str)
        existing_grid = (
            st.session_state["grid"]
            if "grid" in st.session_state
            else preset_grids["Part 1"][:rows, :cols]
        )
        min_rows = min(rows, existing_grid.shape[0])
        min_cols = min(cols, existing_grid.shape[1])
        new_grid[:min_rows, :min_cols] = existing_grid[:min_rows, :min_cols]
        st.session_state["grid"] = new_grid
   
    return st.session_state["grid"]

def display_grid_editor(grid):
    """
    Render the grid editor UI and handle tile clicks to cycle tile types.

    Parameters
    ----------
    grid : np.ndarray
        2D array of strings representing the current grid (e.g., "G", ".", "B", "S", "W").

    Behavior
    --------
    - Injects CSS to create a responsive grid layout and style buttons.
    - Iterates over grid cells and renders a styled button for each cell using
      Streamlit columns and `stylable_container`.
    - Button label shows the current tile symbol and clicking a button cycles the
      tile through `tile_options`, updates `st.session_state["grid"]`, clears
      `st.session_state["mdp_results"]`, and triggers a rerun.

    Notes
    -----
    - Expects the following globals to be defined:
      **color_map** : dict mapping tile symbols to CSS color strings;
      **tile_options** : list of tile symbols in cycle order;
      **stylable_container** : helper for injecting per-button CSS.
    - Uses `st.button` with unique keys per cell to detect clicks.
    - Returns None; updates are applied via `st.session_state`.
    """

    st.markdown(
        f"""
        <style>
        .grid-container {{
            display: grid;
            grid-template-columns: repeat({grid.shape[1]}, 1fr);
            gap: 2px;
            justify-content: center;
            align-items: center;
        }}
        .grid-button {{
            width: 100%;
            height: 50px;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid black;
            border-radius: 5px;
        }}
        .grid-button:hover {{
            opacity: 0.8;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
   
    for r in range(grid.shape[0]):
        row_buttons = st.columns(grid.shape[1], gap="small")
  
        for c in range(grid.shape[1]):
            tile = grid[r, c]
            button_key = f"btn_{r}_{c}"
            tile_color = color_map[tile]
      
            with row_buttons[c]:
                with stylable_container(
                    key=f"container_{r}_{c}",
                    css_styles=f"""
                    button {{
                        background-color: {tile_color};
                        color: black;
                        font-size: 18px;
                        font-weight: bold;
                        border: 2px solid black;
                        border-radius: 5px;
                        width: 100%;
                        height: 50px;
                    }}
                    """,
                ):
                    if st.button(
                        tile,
                        key=button_key,
                        help=f"Click to change tile (Current: {tile})",
                        use_container_width=True,
                    ):
                        next_index = (tile_options.index(tile) + 1) % len(tile_options)
                        grid[r, c] = tile_options[next_index]
                        st.session_state["grid"] = grid
                        st.session_state["mdp_results"] = None
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)