"""
This module prepares Navbar of Gateway UI
"""

from nicegui import ui, events


def highlight_if_route(route: str, current_route: str) -> str:
    """
    Set navbar item highlighted if it is current path
    :param route:
    :param current_route:
    :return:
    """
    if route == current_route:
        return "unelevated color=blue text-white"
    return "flat color=white"


def navbar():
    """
    This method prepares navbar
    :return:
    """

    def on_upload_import_query_result(e: events.UploadEventArguments):
        text = e.content.read().decode("utf-8")
        selected_engine = engine_select.value
        if selected_engine == "Duck":
            pass
        elif selected_engine == "MySQL":
            pass
        elif selected_engine == "PostgreSQL":
            pass
        else:
            ui.notify("Invalid engine selected")

    current_route = ui.context.client.page.path

    with ui.dialog() as dialog, ui.card():
        ui.label("Test")
        engine_list = ["Duck", "MySQL", "Postgres"]
        engine_select = ui.select(options=engine_list, value=engine_list[0])
        ui.upload(on_upload=on_upload_import_query_result)

    with ui.header().classes("items-center justify-between p-0 px-4 no-wrap"):
        with ui.row().classes("items-center"):
            ui.label("Query Profiler").classes("text-xl font-bold")
            ui.separator().props("vertical")
            with ui.row():
                ui.button("Query", on_click=lambda: ui.navigate.to("/")).props(highlight_if_route("/", current_route))
                ui.button("Analyze", on_click=lambda: ui.navigate.to("/analyze")).props(
                    highlight_if_route("/analyze", current_route)
                )
                ui.button("Import Query Result", on_click=dialog.open).props("flat color=white")

