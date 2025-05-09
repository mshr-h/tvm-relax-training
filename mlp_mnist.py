import torch
import torchvision
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.training import SetupTrainer, Trainer
from tvm.relax.training.loss import CrossEntropyLoss
from tvm.relax.training.optimizer import Adam


n_epochs = 5
lr = 0.001
batch_size = 8

train_dataset = torchvision.datasets.MNIST(
    root="~/data/dataset",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = torchvision.datasets.MNIST(
    root="~/data/dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


class RelaxMLP(nn.Module):
    def __init__(self):
        super(RelaxMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def backbone(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.softmax(x, axis=1)
        return x


# tvm training
tvm_mod, params = RelaxMLP().export_tvm(
    {"backbone": {"x": nn.spec.Tensor((batch_size, 784), "float32")}}
)
tvm_mod = tvm_mod.with_attrs({"param_num": len(params), "state_num": 0})
func = tvm_mod["backbone"]

pred_sinfo = func.ret_struct_info
trgt_sinfo = relax.TensorStructInfo((batch_size,), "int64")

setup_trainer = SetupTrainer(
    CrossEntropyLoss(reduction="sum"),
    Adam(lr=lr),
    [pred_sinfo, trgt_sinfo],
)

tvm_target = tvm.target.Target("llvm -mcpu=skylake-avx512")
dev = tvm.cpu(0)

train_mod = setup_trainer(tvm_mod)
ex = tvm.compile(train_mod, tvm_target)
vm = relax.VirtualMachine(ex, dev)

trainer = Trainer(train_mod, vm, dev, False)
trainer.xaiver_uniform_init_params()

for epoch in range(1, n_epochs + 1):
    # train
    for idx, (data, target) in enumerate(train_loader):
        data = data.reshape(-1, 28 * 28)
        trainer.update(data, target)

    # test
    test_loss = 0.0
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data = data.reshape(-1, 28 * 28)
        output = trainer.predict(data)
        pred = torch.from_dlpack(output.numpy().argmax(axis=1, keepdims=True))
        correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        f"TVM [{epoch}] Test set: Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )


class TorchMLP(torch.nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


# torch training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TorchMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.NLLLoss(reduction="sum")

for epoch in range(1, n_epochs + 1):
    # train
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 28 * 28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # test
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"Torch [{epoch}] Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )
