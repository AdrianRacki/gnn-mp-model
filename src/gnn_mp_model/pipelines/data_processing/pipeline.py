"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_data,
    preprocess_ilt_database,
    preprocess_main_database,
    random_data_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_ilt_database,
                inputs="ilt_database",
                outputs="preprocessed_ilt_database",
                name="preprocess_ilt_database",
            ),
            node(
                func=preprocess_main_database,
                inputs="main_database",
                outputs="preprocessed_main_database",
                name="preprocess_main_database",
            ),
            node(
                func=merge_data,
                inputs=["preprocessed_main_database", "preprocessed_ilt_database"],
                outputs="merged_database",
                name="merge_data",
            ),
            node(
                func=random_data_split,
                inputs=["merged_database", "params:split_ratio"],
                outputs=["random_train", "random_test"],
                name="random_data_split",
            ),

        ]
    )
