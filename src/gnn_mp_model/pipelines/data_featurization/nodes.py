"""
This is a boilerplate pipeline 'data_featurization'
generated using Kedro 0.19.6
"""
import typing as t

import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from torch_geometric.loader import DataLoader

from gnn_mp_model.pipelines.data_featurization.utils import from_smiles

transform = T.AddRandomWalkPE(walk_length=30, attr_name="pe")


def _generate_graph_list(df: pd.DataFrame) -> t.List:
    data_list = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        label = row["MP"]
        graph = from_smiles(smiles)
        graph.y = label
        graph = transform(graph)
        smiles_list = smiles.split(".")
        if "+" in smiles_list[0]:
            cation_smiles = smiles_list[0]
            anion_smiles = smiles_list[1]
        else:
            cation_smiles = smiles_list[1]
            anion_smiles = smiles_list[0]
        cation = Chem.MolFromSmiles(cation_smiles)
        anion = Chem.MolFromSmiles(anion_smiles)
        cation_wt = MolWt(cation)
        anion_wt = MolWt(anion)
        cation_natoms = len(cation.GetAtoms())
        cation_nbonds = len(cation.GetBonds())
        anion_natoms = len(anion.GetAtoms())
        anion_nbonds = len(anion.GetBonds())
        graph.gf = torch.tensor(
            [
                cation_wt,
                cation_natoms,
                cation_nbonds,
                anion_wt,
                anion_natoms,
                anion_nbonds,
            ]
        ).unsqueeze(0)
        data_list.append(graph)
    return data_list


# Nodes
def generate_graph_loader(
    df: pd.DataFrame, train: bool, batch_size: int
) -> torch_geometric.loader.DataLoader:
    data_list = _generate_graph_list(df)
    graph_loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
    )
    return graph_loader
