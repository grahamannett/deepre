import reflex as rx

from deepre.state import LLMInfo, ResearchState


def model_settings_tab(llm_info: LLMInfo) -> rx.Component:
    _, llm_info = llm_info[0], llm_info[1]

    def _change(f: str):
        def _fn(value: str) -> None:
            return ResearchState.on_change_model_field_setting(llm_info.typeof, f, value)

        return _fn

    return rx.tabs.content(
        rx.card(
            rx.form(
                rx.hstack(
                    rx.heading(llm_info.typeof),
                    rx.button(
                        "update",
                        type="submit",
                    ),
                ),
                rx.vstack(
                    rx.input(
                        value=llm_info.model_name,
                        name="model_name",
                        on_change=_change("model_name"),
                    ),
                    rx.input(
                        value=llm_info.base_url,
                        name="base_url",
                        on_change=_change("base_url"),
                    ),
                    rx.input(
                        value=llm_info.api_key,
                        name="api_key",
                        on_change=_change("api_key"),
                    ),
                    class_name="p-4",
                ),
                on_submit=lambda data: ResearchState.on_submit_model_setting(llm_info.typeof, data),
            ),
        ),
        value="model_settings",
        padding="1em",
    )


def settings_component() -> rx.Component:
    return rx.flex(
        rx.foreach(ResearchState.models, model_settings_tab),
    )
