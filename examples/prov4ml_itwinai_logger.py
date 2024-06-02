import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

# start the run in the same way as with mlflow
logger = prov4ml.ProvMLItwinAILogger()
logger.create_logger_context()

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
  
    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        logger.log(item=loss.item(), identifier="MSE", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        logger.log(item=loss.item(), identifier="MSE", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.VALIDATION, step=self.current_epoch)        
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        logger.log(item=loss.item(), identifier="MSE", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.EVALUATION, step=self.current_epoch)
        return loss
    
    def on_train_epoch_end(self) -> None:
        logger.log(item=self.current_epoch, identifier="epoch", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        logger.log(item=self, identifier=f"model_version_{self.current_epoch}", kind=prov4ml.LoggingItemKind.MODEL_VERSION, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        logger.log(item=None, identifier="system", kind=prov4ml.LoggingItemKind.SYSTEM_METRIC, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        logger.log(item=None, identifier="carbon", kind=prov4ml.LoggingItemKind.CARBON_METRIC, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        logger.log(item=None, identifier="execution_time", kind=prov4ml.LoggingItemKind.EXECUTION_TIME, context=prov4ml.Context.TRAINING, step=self.current_epoch)
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        logger.log(item=optim, identifier="optimizer", kind=prov4ml.LoggingItemKind.PARAMETER)
        return optim


mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
logger.log(item=tform, identifier="dataset_transformation", kind=prov4ml.LoggingItemKind.PARAMETER)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
val_ds = Subset(train_ds, range(BATCH_SIZE * 1))
train_ds = Subset(train_ds, range(BATCH_SIZE * 2))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    logger=[],
    enable_checkpointing=False, 
)

trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)
# log final version of the model 
# it also logs the model architecture as an artifact by default
logger.log(item=mnist_model, identifier="model_version_final", kind=prov4ml.LoggingItemKind.MODEL_VERSION)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
result = trainer.test(mnist_model, test_loader)

logger.destroy_logger_context()