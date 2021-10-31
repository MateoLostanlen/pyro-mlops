from google_drive_downloader import GoogleDriveDownloader as gdd
from pyrovision.models.rexnet import rexnet1_0x
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import json


# Download dataset
gdd.download_file_from_google_drive(file_id='1GudZUtTgDyhqxqZmF3-p-f645iMJeqRb',
                                    dest_path='./ds.zip',
                                    unzip=True)


# Init
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tf = transforms.Compose([transforms.Resize(size=(448)), transforms.CenterCrop(size=224),
                         transforms.ToTensor(), normalize])

model = rexnet1_0x(pretrained=True).eval()

ds_val = ImageFolder(root='ardeche_val', transform=tf)
val_loader = DataLoader(ds_val, batch_size=4)

ok_fire, ok_no_fire, num_samples_fire, num_samples_no_fire = 0, 0, 0, 0
for x, target in val_loader:

    # Forward
    out = model(x)

    # Apply sigmoid
    out = torch.sigmoid(out)

    samples_fire = target >= 0.5
    samples_no_fire = target >= 0.5

    num_samples_fire += int(torch.sum(samples_fire).item())
    num_samples_no_fire += int(torch.sum(samples_no_fire).item())

    ok_fire += int(torch.sum(out[samples_fire] >= 0.5).item())
    ok_no_fire += int(torch.sum(out[samples_no_fire] < 0.5).item())


accuracy_fire = ok_fire/num_samples_fire
accuracy_no_fire = ok_no_fire/num_samples_no_fire
accuracy = (ok_fire + ok_no_fire)/(num_samples_fire + num_samples_no_fire)

# Write results to file
with open("test_score.json", 'w') as outfile:
    json.dump({"accuracy": accuracy, "accuracy_fire": accuracy_fire, "accuracy_no_fire": accuracy_no_fire}, outfile)
