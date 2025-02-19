import reflex as rx

from deepre.components.general import final_report_component, index_header
from deepre.components.responses_component import (
    logs_component,
    research_input_component,
    responses_component,
)
from deepre.components.settings_component import settings_component


def index_content() -> rx.Component:
    return rx.container(
        rx.vstack(
            index_header(),
            research_input_component(),
            final_report_component(),
            # Logs Section
            # responses_component(),
            logs_component(),
            spacing="4",
            width="100%",
            max_width="1200px",
        ),
        padding="2rem",
    )


def index() -> rx.Component:
    return rx.container(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("main", value="main"),
                rx.tabs.trigger("settings", value="model_settings"),
                width="100%",
            ),
            rx.tabs.content(index_content(), value="main"),
            rx.tabs.content(settings_component(), value="model_settings"),
            default_value="model_settings",  # helpful to change this to see a tab
            # default_value="main",
            variant="enclosed",
            width="100%",
        ),
    )
