# dags/nyc_train_taskflow.py
import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

@dag(
    dag_id="nyc_train_taskflow",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),  # pick how far back you want backfill to go
    schedule="@monthly",
    catchup=True,       # <-- allow historical runs to be created
    tags=["ml", "xgboost", "mlflow"],
)
def pipeline():
    @task
    def train():
        # Import inside the task (keeps parsing fast)
        from lib.duration_prediction_new import run

        ctx = get_current_context()
        ld: pendulum.DateTime = ctx["logical_date"]  # the start of this run's data interval
        # Compute the TRAIN month = logical_date - 2 months
        train_dt = ld.subtract(months=2)
        year = int(train_dt.strftime("%Y"))
        month = int(train_dt.strftime("%m"))

        # Your run() will automatically use (year, month+1) as VALIDATION
        return run(year, month)

    train()

pipeline()
