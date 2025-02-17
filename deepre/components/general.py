import reflex as rx

from deepre.state import ResearchState


def final_report_component() -> rx.Component:
    return rx.cond(
        ResearchState.final_report,
        rx.box(
            rx.vstack(
                rx.heading("Final Report", size="4"),
                rx.markdown(ResearchState.final_report),
                spacing="2",
            ),
            width="100%",
            padding="1rem",
            border="1px solid #e0e0e0",
            border_radius="lg",
            margin_top="1rem",
        ),
    )


def index_header() -> rx.Component:
    return rx.heading(
        "Open Deep Researcher 🔬",
        size="8",
        margin_bottom="1rem",
        margin_left="16rem",
    )
