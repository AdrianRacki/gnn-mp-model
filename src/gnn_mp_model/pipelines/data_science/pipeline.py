"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "params:model_params",
                    "random_train_dataloader",
                    "random_test_dataloader",
                ],
                outputs=["GNN_model", "GNN_model_local"],
                name="train_model",
            ),
        ]
    )
