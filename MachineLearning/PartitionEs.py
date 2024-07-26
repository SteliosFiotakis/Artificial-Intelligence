import cv2

tr_cats_df = []
tr_dogs_df = []

for i in range(1, 4001):
    temp_cat = cv2.imread(f'./CatsAndDogs/training_set/cats/cat.{str(i)}.jpg')
    tr_cats_df.append(temp_cat)
    temp_dog = cv2.imread(f'./CatsAndDogs/training_set/dogs/dog.{str(i)}.jpg')
    tr_dogs_df.append(temp_dog)

te_cats_df = []
te_dogs_df = []

for i in range(4001, 5001):
    temp_cat = cv2.imread(f'./CatsAndDogs/test_set/cats/cat.{str(i)}.jpg')
    te_cats_df.append(temp_cat)
    temp_dog = cv2.imread(f'./CatsAndDogs/test_set/dogs/dog.{str(i)}.jpg')
    te_dogs_df.append(temp_dog)