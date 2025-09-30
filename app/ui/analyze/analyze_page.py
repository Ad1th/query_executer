from nicegui import ui, events
import json

from app.sampling_methods.adaptive_balanced_sampling import sample_adaptive_balanced
from app.sampling_methods.calculate_qerr import fit_polynomial_on_sample, predict_and_qerr_for_all, summarize_qerr
from app.sampling_methods.stratified_time_sampling import sample_stratified
from app.ui.analyze.helpers import load_runtime_from_json, extract_filters


def plot_qerr(evaluation_df, filter_name: str):
    x = evaluation_df[filter_name]
    y = evaluation_df['qerr']

    with ui.matplotlib(figsize=(6, 4)).figure as fig:
        ax = fig.gca()
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel(filter_name)
        ax.set_ylabel('Q-error')
        ax.set_title(f'Q-error vs {filter_name}')
        ax.grid(True, linestyle='--', alpha=0.6)


sampling_method_map = {
    'Adaptive': sample_adaptive_balanced,
    'Stratified': sample_stratified,
}

upload_table_columns = [
    {'name': 'id', 'label': 'Id', 'field': 'id', 'required': True},
    {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True},
]

upload_table_rows = []
upload_list = []


def analyze_page():
    # TODO: Read filters from first entry
    # TODO: Separate Upload and current query tables. Current tables must be fetched from BackendService.ResultStorage
    #       Maybe single table better ?
    def on_upload_query_result(e: events.UploadEventArguments):

        text = e.content.read().decode("utf-8")
        data_list = json.loads(text)
        upload_name = input_box.value
        upload_list_id = len(upload_list)
        upload_table.add_row({"id": upload_list_id, "name": upload_name})
        upload_list.append(data_list)

    def on_click_calculate_qerr():
        row_id = upload_table.selected[0].get("id")
        dataset = upload_list[row_id]
        filter = filters_select.value
        runtime = load_runtime_from_json(dataset, filter)

        method_value = sampling_method_select.value
        sampling_method = sampling_method_map[method_value]
        sampling_result = sampling_method(runtime)

        model = fit_polynomial_on_sample(sampling_result, filter_name=filter)
        evaluation = predict_and_qerr_for_all(runtime, model, filter_name=filter, engine="auto")

        plot_qerr(evaluation, filter_name=filter)

        pass

    def on_select_upload_table():
        """
        When selected row changes in UploadTable, update filters with new row filters
        """
        row_id = upload_table.selected[0].get("id")
        dataset = upload_list[row_id]
        filters = extract_filters(dataset[0])
        #ilters_select.set_options(filters, value=1)
        #filters_select.update()

    @ui.refreshable
    def generate_plots():
        pass

    with ui.row():
        with ui.column():
            ui.label("Upload JSON Data")
            ui.label("Sample, Fit and Calculate Qerr")
            input_box = ui.input(value="Default")
            ui.upload(on_upload=on_upload_query_result).classes('w-[200px]')

        with ui.column():
            upload_table = ui.table(columns=upload_table_columns, rows=upload_table_rows, row_key='id', on_select=on_select_upload_table)
            upload_table.set_selection("multiple")
            sampling_method_select = ui.select(options=["Stratified", "Adaptive", "Random", "Uniform"], multiple=True).props('use-chips')
        #filters_select = ui.select(options=[])
        with ui.column():
            ui.button("Sampling QError", on_click=on_click_calculate_qerr)
            ui.button("Engine QError", on_click=on_click_calculate_qerr)
