"""
This is a boilerplate pipeline 'data_featurization'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_graph_loader


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_graph_loader,
                inputs=["random_test", "params:test", "params:test_batch_size"],
                outputs="random_test_dataloader",
                name="generate_graph_loader_rt2",
            ),
            node(
                func=generate_graph_loader,
                inputs=["random_train", "params:train", "params:train_batch_size"],
                outputs="random_train_dataloader",
                name="generate_graph_loader_rt1",
            ),
            node(
                func=generate_graph_loader,
                inputs=["merged_database", "params:test", "params:predict_batch_size"],
                outputs="predict_dataloader",
                name="generate_predict_loader",
            ),
        ]
    )
