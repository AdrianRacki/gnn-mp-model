"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.6
"""
import lightning as L
import mlflow
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.nn import (
    BatchNorm1d,
    L1Loss,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import (
    GINEConv,
    GPSConv,
    GraphNorm,
    SAGPooling,
    SetTransformerAggregation,
)


# class nn.module
class GNN(torch.nn.Module):
    def __init__(
        self, hidden_size: int, dense_size: int, num_layers: int, pooling: bool
    ):  # noqa: PLR0913
        # Loading params
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        node_dim = 9
        edge_dim = 4
        pe_dim = 8
        pool_rate = 0.75
        # Initial embeddings
        self.node_emb = Linear(pe_dim + node_dim, hidden_size)
        self.pe_lin = Linear(30, pe_dim)
        self.pe_norm = BatchNorm1d(30)
        self.edge_emb = Linear(edge_dim, hidden_size)
        self.aggr = SetTransformerAggregation(hidden_size)
        # Module lists
        self.gps_list = ModuleList([])
        self.gn_list = ModuleList([])
        self.aggr_list = ModuleList([])
        self.pool_list = ModuleList([])
        # Lists generation
        for _ in range(self.num_layers):
            nn = Sequential(
                Linear(hidden_size, hidden_size),
                ReLU(),
                Linear(hidden_size, hidden_size),
            )
            self.gps_list.append(
                GPSConv(
                    hidden_size,
                    GINEConv(nn, edge_dim=hidden_size),
                    heads=4,
                    dropout=0.2,
                )
            )
            self.gn_list.append(GraphNorm(hidden_size))
            self.aggr_list.append(SetTransformerAggregation(hidden_size))
            self.pool_list.append(SAGPooling(hidden_size, pool_rate))

        # Linear output layers
        self.linear1 = Linear(hidden_size, dense_size)
        self.linear2 = Linear(dense_size, int(dense_size / 2))
        self.linear3 = Linear(int(dense_size / 2), int(dense_size / 4))
        self.linear4 = Linear(int(dense_size / 4), 1)

    def forward(self, x, pe, edge_attr, edge_index, batch_index, gf):  # noqa: PLR0913
        # Initial embeddings
        x_pe = self.pe_norm(pe)
        x = torch.cat((x, self.pe_lin(x_pe)), 1)
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        global_representation = []
        global_representation.append(self.aggr(x, batch_index))
        # Internal convolutions
        for i in range(self.num_layers):
            x = self.gps_list[i](x, edge_index, batch_index, edge_attr=edge_attr)
            x = self.gn_list[i](x, batch_index)
            if self.pooling is True:
                x, edge_index, edge_attr, batch_index, _, _ = self.pool_list[i](
                    x, edge_index, edge_attr, batch_index
                )
            global_representation.append(self.aggr_list[i](x, batch_index))
        # Output block
        x = sum(global_representation)
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.relu(self.linear3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear4(x)
        return x


class GNN_L(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = GNN(
            params["hidden_size"],
            params["dense_size"],
            params["num_layers"],
            params["pooling"],
        )
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.gamma = params["gamma"]
        self.loss_fn = L1Loss()
        self.save_hyperparameters(params)

    def forward(self, x, pe, edge_attr, edge_index, batch_index, gf):  # noqa: PLR0913
        return self.model(x.float(), pe, edge_attr.float(), edge_index, batch_index, gf)

    def training_step(self, batch, batch_index):
        preds = self(
            batch.x.float(),
            batch.pe,
            batch.edge_attr.float(),
            batch.edge_index,
            batch.batch,
            batch.gf.float(),
        ).squeeze()
        target = batch.y.float()
        loss = self.loss_fn(preds, target)
        r2 = r2_score(target.numpy(), preds.detach().numpy())
        rmse = root_mean_squared_error(target.numpy(), preds.detach().numpy())
        self.log("r2", r2)
        self.log("rmse", rmse)
        self.log("mae", loss)
        return loss

    def validation_step(self, batch, batch_index):
        preds = self(
            batch.x.float(),
            batch.pe,
            batch.edge_attr.float(),
            batch.edge_index,
            batch.batch,
            batch.gf.float(),
        ).squeeze()
        target = batch.y.float()
        val_loss = self.loss_fn(preds, target)
        val_r2 = r2_score(target.numpy(), preds.detach().numpy())
        val_rmse = root_mean_squared_error(target.numpy(), preds.detach().numpy())
        self.log("val_mae", val_loss)
        self.log("val_r2", val_r2)
        self.log("val_rmse", val_rmse)

    def test_step(self, batch, batch_index, dataloader_idx=0):
        preds = self(
            batch.x.float(),
            batch.pe,
            batch.edge_attr.float(),
            batch.edge_index,
            batch.batch,
            batch.gf.float(),
        ).squeeze()
        target = batch.y.float()
        test_mae = self.loss_fn(preds, target)
        test_r2 = r2_score(target.numpy(), preds.detach().numpy())
        self.log("test_mae", test_mae)
        self.log("test_r2", test_r2)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = [
            self(
                batch.x.float(),
                batch.pe,
                batch.edge_attr.float(),
                batch.edge_index,
                batch.batch,
                batch.gf.float(),
            ).squeeze(),
            batch.smiles,
            batch.y,
        ]
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(  # noqa: F811
            optimizer, gamma=self.gamma
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def train_model(params, train_dataloader, test_dataloader):
    filename = "GPS_Main_Model"
    L.seed_everything(47)
    model = GNN_L(params)
    early_stopping = EarlyStopping("val_mae", patience=10, mode="min", strict=False)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_mae:.2f}-{val_rmse:.2f}",
        monitor="val_mae",
        save_top_k=2,
        mode="min",
    )

    logger = CSVLogger(save_dir="lightning_logs", name=filename)
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        log_every_n_steps=20,
        logger=logger,
        deterministic=True,
        accumulate_grad_batches=1,
    )
    # Model pretraining
    mlflow.pytorch.autolog()
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )
    return model, model
