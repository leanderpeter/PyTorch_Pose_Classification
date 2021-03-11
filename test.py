from preprocessing import preprocessing

path = './test_dataset'

dataset = preprocessing().process_dataset(path)


img_path = './pose/dataset/Left/gesture_ - 45.jpeg'
img = preprocessing().img_read(img_path)