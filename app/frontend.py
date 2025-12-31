import json

from fastapi import FastAPI
from nicegui import ui, events

from app.config import load_config
from app.backend_service import BackendService, start_db_connections, get_min_max_of_column
from app.helpers import extract_variables
from app.models import BenchmarkQuery
from app.ui.analyze.analyze_page import analyze_page
from app.ui.common.navbar import navbar

config = load_config()

backend_service = BackendService()


result_table_columns = [
    {'name': 'id', 'label': 'Id', 'field': 'id', 'required': True},
    {'name': 'server', 'label': 'Server', 'field': 'server', 'required': True},
    {'name': 'database', 'label': 'Database', 'field': 'database', 'required': True},
    {'name': 'query', 'label': 'Query', 'field': 'query', 'required': True},
    {'name': 'download', 'label': 'Download', 'field': 'download', 'required': True},
]

result_table_rows = []

benchmark_query_list = []

query_table_columns = [
    {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True},
    {'name': 'database', 'label': 'Database', 'field': 'database', 'required': True},
    {'name': 'benchmark', 'label': 'Benchmark', 'field': 'benchmark', 'required': True},
]

query_table_rows = []

batch_results = []
total_executed_batch = 0

@ui.page("/")
async def main_page():
    navbar()
    await backend_service.initialize_queue_worker()
    queries_in_queue = 0
    db_clients_local = start_db_connections()

    def result_table_update(benchmark_query: BenchmarkQuery):
        global total_executed_batch
        result_table.add_row(
            {
                'id': total_executed_batch,
                'server': benchmark_query.database,
                'database': benchmark_query.benchmark,
                'query': benchmark_query.name,
            }
        )
        total_executed_batch += 1
        queue_information.refresh(0, 0, False)

    def on_click_save_query():
        """
        Save current query on SQL editor into query table
        """
        query = sql_editor.value
        parameters = extract_variables(query)
        print(parameters)
        benchmark_query = BenchmarkQuery(
            query=query,
            parameters=parameters,
            database=dropdown_db.value,
            benchmark=dropdown_bm.value,
            name=name_input.value
        )
        benchmark_query_list.append(benchmark_query)
        add_query_to_table(benchmark_query.to_dict())

    def add_query_to_table(new_query):
        """
        Add a new query to the table
        """
        if any(query['name'] == new_query['name'] for query in query_table.rows):
            ui.notify(f"Query with name {new_query['name']} already exists")
            return

        query_table.add_row(new_query)

    def update_code_block():
        """
        Updates SQL code inspection block
        """
        code_block.content = query_table.selected[0]["query"]
        code_block.update()
        variable_parameters.refresh()

    @ui.refreshable
    def variable_parameters():
        async def on_click_start_query_execution():
            """
            Executes selected query iterating over parameter range values
            """
            nonlocal queries_in_queue
            range_values = []
            for handle in var_input_handles:
                range_value_string = handle.value
                range_value = tuple(int(x.strip()) for x in range_value_string.split(","))
                range_values.append({'name': handle.label, 'range': range_value, 'type': 'INT'})
            query_template = query_table.selected[0]
            benchmark_query = BenchmarkQuery.from_dict(query_template)
            await backend_service.schedule_query_exectution(benchmark_query, range_values)
            print("Query added to queue")
            queries_in_queue += 1
            queue_information.refresh(0, 0, False)

        with ui.card():
            ui.label("Range Values for Parameters")
            ui.label("Example: For a variable starting from 10 to 100 with increment 5 -> '10,100,5'")
            ui.label("For fixed value, use a single value like '10'")
            if query_table.selected:
                var_input_handles = []
                variables = query_table.selected[0]["parameters"]
                for variable in variables:
                    with ui.column():
                        var_name = variable.get("name")
                        # TODO: Implement support for Postgres and DuckDB here
                        min_value, max_value = get_min_max_of_column(db_clients_local["MySQL"], var_name)
                        ui.label(
                            f"Range values for Parameter: {var_name}, Type: {variable.get("data_type")}")
                        ui.label(f"Min :{min_value}, Max: {max_value}")
                        var_input = ui.input(label=var_name)
                        var_input_handles.append(var_input)
                ui.button("Start Query Execution", on_click=on_click_start_query_execution)

    def on_click_import_queries():
        """
        Callback for "Import Queries" button
        """
        ui.download.content(json.dumps([bq.to_dict() for bq in benchmark_query_list]))

    def on_upload_import_queries(e: events.UploadEventArguments):
        """
        Callback for "Import Queries" upload block
        """
        text = e.content.read().decode("utf-8")
        data_list = json.loads(text)
        for data in data_list:
            benchmark_query_list.append(BenchmarkQuery.from_dict(data))
            add_query_to_table(data)
        ui.notify("Upload done")

    def on_row_download_result(msg):
        """
        Callback for download button of row
        """
        id = msg.args['key']
        row = msg.args['row']
        server = row["server"]
        db = row["database"]
        q = row["query"]
        ui.download.content(json.dumps(backend_service.result_storage.parsed_result_list[id]),
                            f"{server}_{db}_{q}_{id}.json")
        ui.download.content(json.dumps(backend_service.result_storage.raw_result_list[id]),
                            f"{server}_{db}_{q}_{id}_raw.json")

    @ui.refreshable
    def queue_information(i=0, total=0, executing=False):
        ui.label("Queue Information")
        ui.label(f"Query batch in queue: {queries_in_queue}")
        if executing:
            ui.label(f"Query Completed N:{i+1}/{total}")
        else:
            ui.label("No queries are executing currently")

    backend_service.set_table_update_callback(result_table_update)
    # UI code starts here
    with ui.row():
        with ui.column():
            with ui.grid(columns="300px auto auto auto auto"):
                # SQL Code editor
                with ui.card():
                    sql_editor = ui.codemirror()
                # Database select
                with ui.card():
                    with ui.row():
                        with ui.column():
                            name_input = ui.input(label="Query Name")
                            db_list = list(db_clients_local.keys())
                            ui.label("Server")
                            dropdown_db = ui.select(options=db_list, label="Server",
                                                    value=db_list[0])
                            benchmark_list = ["TPC-H 1GB", "TPC-H 10GB", "TPC-DS 1GB", "TPC-DS 10GB"]
                            dropdown_bm = ui.select(options=benchmark_list, label="Benchmark", value=benchmark_list[0])
                            ui.button(text="Save Query", on_click=on_click_save_query)
                        with ui.column():
                            ui.button("Delete Selected")
                            ui.button("Export Queries", on_click=on_click_import_queries)
                            ui.label("Import Queries")
                            ui.upload(on_upload=on_upload_import_queries).classes('w-[200px]')
                # Query template table
                query_table = ui.table(rows=query_table_rows, columns=query_table_columns, row_key='name', on_select=update_code_block)
                query_table.set_selection('single')
                # Query inspection
                with ui.card():
                    ui.label("Inspect Selected Query")
                    code_block = ui.code(language="sql").classes('w-full')
                # Variable values
                variable_parameters()

            # Query results table
            with ui.row():
                with ui.card():
                    ui.label("Query Execution Results")
                    result_table = ui.table(columns=result_table_columns, rows=result_table_rows, row_key='id')
                    result_table.add_slot(
                        "body-cell-download",
                        """
                        <q-td :props="props">
                            <q-btn @click="$parent.$emit('action', props)" icon="download" flat />
                        </q-td>
                    """,
                    )
                    result_table.on("action", on_row_download_result)
                with ui.card():
                    queue_information()


@ui.page("/analyze")
async def analyze_page_route():
    navbar()
    analyze_page()


def init(fastapi_app: FastAPI) -> None:
    """
    Initialize nicegui with a given FastAPI object
    :param fastapi_app:
    :return:
    """
    ui.run_with(
        fastapi_app,
    )