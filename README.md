# GNN MP model
**This repository is under development.**

Final results will be presented (and added hereafter) at the PTChem2024 scientific conference in September.

## Overview

This is a Kedro project with Kedro-Viz setup and kedro-mlflow plugin for tracking purposes, generated using `kedro 0.19.6`.
These repositories are part of a doctoral thesis aimed at creating models to predict the properties of ionic liquids using graph-level prediction (graph neural networks). These specific focuses predict the melting point. 

The neural network used uses state-of-the-art GNNs, including (GPS) Graph Transformer layers [https://arxiv.org/abs/2205.12454], Adaptive Readout layers [https://arxiv.org/abs/2211.04952], Hierarchical GNNs Architecture [https://arxiv.org/abs/1811.01287] and Graph Normalization Layers [https://arxiv.org/abs/2009.03294].

Project structure generated with kedro-viz:
![kedro-pipeline (3)](https://github.com/user-attachments/assets/1bbdef6a-9962-4e9f-9b7f-e972c0da34f6)



