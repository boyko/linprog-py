import helpers
import ipywidgets as widgets
from IPython.display import clear_output

s1 = """
max 2x1 + 1x2
-1x1 + 2x2 <= 20
2x1 - x2 <= 6
3x1 + 8x2  <= 24
"""

model_input = widgets.Textarea(
    value=s1,
    rows=10,
    continuous_update=True,
    placeholder='Type something',
    description='String:',
    disabled=False
)

run_button = widgets.Button(
    description='Run model',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Run',
    icon='check'  # (FontAwesome names without the `fa-` prefix)
)


@run_button.on_click
def run_model(g):
    clear_output()
    render_ui()
    print("Running model")
    print(model_input.value)

    helpers.parse_eq(model_input.value)


@run_button.on_click
def run_model(g):
    clear_output()
    render_ui()
    print("Running model")
    print(model_input.value)

    helpers.parse_eq(model_input.value)


def render_ui():
    display(model_input)
    display(run_button)


def render_graphical_ui():
    display(model_input)
    display(run_button)
