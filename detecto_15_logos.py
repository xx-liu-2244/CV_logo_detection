from detecto import core, utils
from torchvision import transforms

labels = ['Nike', 'Adidas', 'Under Armour', 'Puma', 'The North Face',
           'Starbucks', 'Apple Inc.',
           'Mercedes-Benz', 'NFL', 'Emirates', 'Coca-Cola', 'Chanel',
           'Hard Rock Cafe', 'Toyota', 'Pepsi','Other']

augmentations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(950),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=1.4),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

dataset = core.Dataset(label_data='annot_train.csv',image_folder='train_detecto/',transform=augmentations)
loader = core.DataLoader(dataset, batch_size=3, shuffle=True)
model = core.Model(classes=labels)

model.fit(dataset, epochs=10, learning_rate=0.01,verbose=True)

model.save('detecto_weights_15logos.pth')
