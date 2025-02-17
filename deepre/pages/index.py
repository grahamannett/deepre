import reflex as rx

from deepre.components.responses_component import (
    responses_component,
    research_input_component,
    logs_component,
)

from deepre.components.general import final_report_component, index_header


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            index_header(),
            rx.text("Enter your research query to generate a report."),
            # Input Section
            research_input_component(),
            # Results Section
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
