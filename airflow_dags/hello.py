import pendulum
from airflow.decorators import dag, task  # TaskFlow API

@dag(
    dag_id="hello_taskflow_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["example"],
)
def _hello():
    @task
    def say_hello():
        print("Hello, world!")
    say_hello()

hello = _hello()
