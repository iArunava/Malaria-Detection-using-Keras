# Malaria-Detection-using-Keras

This project uses Keras to detect Malaria from Images. The model used is a ResNet50 which is trained from scratch.
The images in this dataset is divided into to categories
- Parasitized
- Uninfected

## How to use

1. Clone the repository
```
git clone https://github.com/iArunava/Malaria-Detection-using-Keras.git
```

2. cd to the directory
```
cd Malaria-Detection-using-Keras/
```

3. Get some images to infer upon
```
chmod u+x ./datasets/download.sh
./datasets/download.sh
```

4. Find an image of your choice and infer!!
```
python3 train_model.py -i ./path/to/image
```

5. Can even infer on a set of images in `datasets/cimages_test/`
```
python3 train_model.py -otb True
```

6. To see all the options
```
python3 train_model.py --help
```

7.**Enjoy!!**

## A few examples to visualize

![dl_medical_imaging_malaria_dataset](https://user-images.githubusercontent.com/26242097/50046086-713da980-00c3-11e9-9c79-db215df220e2.jpg)

## LICENSE

The code in this repository is distributed under the MIT License. <br/>
Feel free to fork and try it on your own!
