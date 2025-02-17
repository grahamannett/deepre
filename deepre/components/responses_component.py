import reflex as rx

from deepre.state import ResearchState, Response


def responses_card(response: Response, index_id: str) -> rx.Component:
    return rx.box(
        rx.heading(f"ID: {index_id}"),
        rx.markdown(response.response_text),
        border_radius="lg",
        # margin_top="1rem",
        # padding="1rem",
        padding="0.5rem",
        overflow_y="auto",
        border="1px solid #e0e0e0",
        id=response.response_id,
        # id=f"response-card-id-{index_id}",
    )


def responses_component() -> rx.Component:
    return rx.scroll_area(
        rx.foreach(
            ResearchState.responses,
            responses_card,
        ),
        # rx.box(id=ResearchState.responses_box_id),
        # style={"height": 250},
        width="100%",
        height="250px",
        padding="0.5rem",
        border="1px solid #e0e0e0",
        border_radius="lg",
        # margin_top="3rem",
        overflow_y="auto",
    )


def submit_button() -> rx.Component:
    # can use this for testing
    # def make_responses() -> rx.Component:
    #     return rx.button("Test Responses", on_click=ResearchState.make_responses)

    return rx.hstack(
        rx.button(
            "Start Research",
            on_click=ResearchState.handle_submit,
            loading=ResearchState.is_processing,
            color_scheme="blue",
        ),
        width="40%",
        align_self="center",
    )


def research_input_component() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.input(
                placeholder="Research Query/Topic",
                value=ResearchState.user_query,
                on_change=ResearchState.set_user_query,
                width="100%",
            ),
            submit_button(),
            spacing="3",
        ),
        width="100%",
        padding="1rem",
        border="1px solid #e0e0e0",
        border_radius="lg",
    )


def logs_component() -> rx.Component:
    return rx.cond(
        ResearchState.process_logs,
        rx.box(
            rx.vstack(
                rx.heading("Process Logs", size="4"),
                rx.markdown(ResearchState.process_logs),
                # spacing="2",
            ),
            width="100%",
            height="200px",
            padding="1rem",
            border="1px solid #e0e0e0",
            border_radius="lg",
            margin_top="1rem",
            overflow_y="auto",
        ),
    )
