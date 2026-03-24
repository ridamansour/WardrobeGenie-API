"""
WardrobeGenie Batch Ingestion Pipeline
======================================
Triggered asynchronously to process heavy user wardrobe uploads.
Extracts YOLOS bounding boxes, crops, and updates the vector DB safely.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from config import PROJECT_ROOT_STR

default_args = {
    'owner': 'wardrobe_genie_admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 22),
    'retries': 1,
}

with DAG(
    'wardrobegenie_batch_ingestion',
    default_args=default_args,
    schedule_interval=None,  # Triggered purely via API
    catchup=False,
    tags=['wardrobegenie', 'mlops', 'production', 'ingestion'],
) as dag:

    start_ingestion = EmptyOperator(task_id='start_ingestion')
    end_ingestion = EmptyOperator(task_id='end_ingestion')

    fetch_user_uploads = BashOperator(
        task_id='fetch_user_uploads',
        bash_command=f"""
            echo "Staging files for Batch ID: {{{{ run_id }}}}" && \
            mkdir -p {PROJECT_ROOT_STR}/data/user_uploads/{{{{ run_id }}}} && \
            python {PROJECT_ROOT_STR}/scripts/fetch_from_s3_or_api.py --batch_id {{{{ run_id }}}}
        """
    )

    run_yolos_detection = BashOperator(
        task_id='run_yolos_detection',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/perception_layer/clothing_detection_yolos && \
            python batch_inference.py --input_dir {PROJECT_ROOT_STR}/data/user_uploads/{{{{ run_id }}}}
        """,
        pool='gpu_pool'
    )

    extract_visual_features = BashOperator(
        task_id='extract_visual_features',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/representation_layer/visual_embeddings && \
            python generate_embeddings.py --input_dir {PROJECT_ROOT_STR}/data/user_uploads/{{{{ run_id }}}}/crops
        """,
        pool='gpu_pool'
    )

    update_vector_database = BashOperator(
        task_id='update_vector_database',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/recomendation_engine && \
            python update_user_wardrobe.py --batch_id {{{{ run_id }}}}
        """
    )

    # Pipeline Flow (Strictly Linear)
    (
        start_ingestion
        >> fetch_user_uploads
        >> run_yolos_detection
        >> extract_visual_features
        >> update_vector_database
        >> end_ingestion
    )