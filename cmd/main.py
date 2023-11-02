import traceback
import warnings
import logging
import mlflow
import click
import os

PATH_ROOT = os.environ['PATH_ROOT'] = "/home/arthur/Documents/ifc/tc/code/sipd-mlops"

os.environ['TEST_STATS_PATH'] = f"{PATH_ROOT}/stats/test"
os.environ['TEST_DATASET_PATH'] = f"{PATH_ROOT}/datasets/test"
os.environ['TRAIN_DATASET_PATH'] = f"{PATH_ROOT}/datasets/train"
os.environ['DATASET_PATH'] = "/home/arthur/Documents/dataset"
os.environ['EXPERIMENT_STATS_PATH'] = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/stats/experiments'
os.environ['EXPERIMENTS_DATASET_PATH'] = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/datasets/experiments'

def _run(entrypoint, parameters=dict(), source_version = None, use_cache = True):
    """Launching new run for an entrypoint"""

    print(
        "Launching new run for entrypoint=%s and parameters=%s"
        % (entrypoint, parameters)
    )
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return submitted_run


@click.command()
def workflow():
    """run the workflow"""

    with mlflow.start_run(run_name="pipeline") as active_run:
        mlflow.set_tag("mlflow.runName", "pipeline")

        # dataset pipeline
        _run("prepare_train_and_test_dataset")
        _run("prepare_experiments_dataset")

        # model pipeline
        _run("train_model")
        _run("test_model")

        # experiments
        _run("experiments")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/train_model.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        workflow()
    except Exception as e:
        print("Exception occured. Check logs.")
        logger.error(f"Failed to run workflow due to error:\n{e}")
        logger.error(traceback.format_exc())