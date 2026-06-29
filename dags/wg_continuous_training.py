"""
WardrobeGenie Continuous Training (CT) Pipeline
===============================================
Automates the fine-tuning of the Triplet Set-Transformer using user feedback.
Conditionally deploys new weights based on evaluation metrics.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

from config import PROJECT_ROOT_STR


def decide_deployment(**kwargs):
    """Evaluates if the newly trained model should be deployed."""
    # In a real environment, read metrics from an evaluation JSON
    new_model_score = 0.85
    old_model_score = 0.82

    if new_model_score > old_model_score:
        return 'deploy_new_weights'
    return 'discard_and_alert'


default_args = {
    'owner': 'wardrobe_genie_admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 22),
}

with DAG(
        'wardrobegenie_continuous_training',
        default_args=default_args,
        schedule_interval='@weekly',
        catchup=False,
        tags=['wardrobegenie', 'mlops', 'training'],
) as dag:
    start = EmptyOperator(task_id='start')

    extract_feedback = BashOperator(
        task_id='extract_feedback',
        bash_command=f"python {PROJECT_ROOT_STR}/src/airflow/scripts/pull_db_logs.py"
    )

    retrain_model = BashOperator(
        task_id='retrain_set_transformer',
        bash_command=f"python {PROJECT_ROOT_STR}/recomendation_engine/train.py",
        pool='gpu_pool'
    )

    evaluate_model = BashOperator(
        task_id='evaluate_metrics',
        bash_command=f"python {PROJECT_ROOT_STR}/src/main/recomendation_engine/evaluate.py"
    )

    check_improvement = BranchPythonOperator(
        task_id='check_improvement',
        python_callable=decide_deployment
    )

    deploy_new_weights = BashOperator(
        task_id='deploy_new_weights',
        bash_command=f"cp {PROJECT_ROOT_STR}/models/new_model.pth {PROJECT_ROOT_STR}/models/production_stylist.pth"
    )

    discard_and_alert = BashOperator(
        task_id='discard_and_alert',
        bash_command="echo 'New model underperformed. Discarding updates.'"
    )

    # Pipeline Flow
    start >> extract_feedback >> retrain_model >> evaluate_model >> check_improvement
    check_improvement >> [deploy_new_weights, discard_and_alert]