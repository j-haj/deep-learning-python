import os, shutil

original_dataset_dir = os.path.join(os.getcwd(),'data/train')
base_dir = original_dataset_dir + '/cats_and_dogs_small'
try:
    os.mkdir(base_dir)
except:
    pass
train_dir = os.path.join(base_dir, 'train')
try:
	os.mkdir(train_dir)
except:
    pass
validation_dir = os.path.join(base_dir, 'validation')
try:
	os.mkdir(validation_dir)
except:
    pass
test_dir = os.path.join(base_dir, 'test')
try:
	os.mkdir(test_dir)
except:
    pass
train_cats_dir = os.path.join(train_dir, 'cats')
try:
    os.mkdir(train_cats_dir)
except:
    pass
train_dogs_dir = os.path.join(train_dir, 'dogs')
try:
    os.mkdir(train_dogs_dir)
except:
    pass
validation_cats_dir = os.path.join(validation_dir, 'cats')
try:
    os.mkdir(validation_cats_dir)
except:
    pass
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
try:
    os.mkdir(validation_dogs_dir)
except:
    pass
test_cats_dir = os.path.join(test_dir, 'cats')
try:
    os.mkdir(test_cats_dir)
except:
    pass
test_dogs_dir = os.path.join(test_dir, 'dogs')
try:
    os.mkdir(test_dogs_dir)
except:
    pass
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('Total training cat images:', len(os.listdir(train_cats_dir)))
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
print('Total test cat images:', len(os.listdir(test_cats_dir)))

print('Total training dog images:', len(os.listdir(train_dogs_dir)))
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('Total test dog images:', len(os.listdir(test_dogs_dir)))
