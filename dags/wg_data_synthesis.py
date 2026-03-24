"""
WardrobeGenie Offline Data Synthesis Pipeline
=============================================
Orchestrates the end-to-end dataset generation for WardrobeGenie.
Handles COCO formatting, zero-shot CLIP pseudo-labeling, and KD targeting.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from config import PROJECT_ROOT_STR, DATA_DIR_STR

default_args = {
    'owner': 'wardrobe_genie_admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 22),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'wardrobegenie_data_synthesis',
    default_args=default_args,
    description='End-to-end data pipeline for WardrobeGenie recommendation engine',
    schedule_interval='@weekly',
    catchup=False,
    tags=['wardrobegenie', 'mlops', 'computer_vision'],
) as dag:

    start_pipeline = EmptyOperator(task_id='start_pipeline')
    end_pipeline = EmptyOperator(task_id='end_pipeline')

    format_coco_dataset = BashOperator(
        task_id='format_coco_dataset',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/perception_layer/clothing_detection_segmentation && \
            python dataset_generation.py --output_dir {DATA_DIR_STR}
        """
    )

    pseudo_label_attributes = BashOperator(
        task_id='pseudo_label_attributes',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/perception_layer/multi_attribute_classifire && \
            python dataset_generation.py \
                --img_dir {DATA_DIR_STR}/train \
                --ann_file {DATA_DIR_STR}/train/_annotations.coco.json \
                --out_dir {DATA_DIR_STR}/attribute_dataset/train \
                --batch_size 64
        """,
        pool='gpu_pool'
    )

    generate_teacher_embeddings = BashOperator(
        task_id='generate_teacher_embeddings',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/representation_layer/visual_embeddings && \
            python generate_embeddings.py
        """,
        pool='gpu_pool'
    )

    build_recommendation_pool = BashOperator(
        task_id='build_recommendation_pool',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/recomendation_engine && \
            python build_dataset.py
        """
    )

    # NLP is independent
    generate_nlp_queries = BashOperator(
        task_id='generate_nlp_queries',
        bash_command=f"""
            cd {PROJECT_ROOT_STR}/semantic_processing/query_vectorization && \
            python generate_queries.py
        """
    )

    # Pipeline Flow
    start_pipeline >> generate_nlp_queries >> end_pipeline
    start_pipeline >> format_coco_dataset
    format_coco_dataset >> [pseudo_label_attributes, generate_teacher_embeddings]
    [pseudo_label_attributes, generate_teacher_embeddings] >> build_recommendation_pool
    build_recommendation_pool >> end_pipeline