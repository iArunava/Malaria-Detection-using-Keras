# Malaria-Detection-using-Keras

This project uses Keras to detect Malaria from Images. The model used is a ResNet50 which is trained from scratch.
The images in this [dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) is divided into to categories
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

## Production

The model is deployed to production and you can use the model to test on your own images!!<br/>
The model is deployed using [Zeit](https://zeit.co/). Quite an amazing platform!<br/>
The live link to the deployed model can be found here: https://malaria-classifier.now.sh <br/>

The code that is used to deploy the model is open sourced and can be found [here](https://github.com/iArunava/Malaria-Detection-using-Keras/tree/master/zeit)

## A look into the deployed model on web

![malaria model - deployed classifier](https://user-images.githubusercontent.com/26242097/50305612-5d29eb80-04b9-11e9-9feb-7c0eb58483c6.png)

## How to deploy your own models using this?

0. Download `node`, `now`, `now-cli`
```
sudo apt install npm
sudo npm install -g now
```

1. Get a **direct download** link to your model

2. Set that link equal to `model_file_url` - which you can find here on [app/server.py/L20](https://github.com/iArunava/Malaria-Detection-using-Keras/blob/master/zeit/app/server.py#L20)

3. Run
```
now
```

4. **The site should be deployed now!!**

5. Use a custom name for your site
```
export NAME='custom-site-name'
now alias $NAME
```
your site is now *also* accessible at **custom-site-name.now.sh**

6. Keeping the deployment alive (as it goes to sleep after some time of inactivity)
```
now scale custom-site-name.now.sh sfo 1
```

7. Share the link with everyone and Enjoy!!


## A few examples to visualize

![dl_medical_imaging_malaria_dataset](https://user-images.githubusercontent.com/26242097/50046086-713da980-00c3-11e9-9c79-db215df220e2.jpg)

## References

1. [PyImageSearch - Deep Learning and Medical Image Analysis with Keras](https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/)
2. [Pre-trained convolutional neural networks as feature extractors toward improved parasite detection in thin blood smear images.](https://lhncbc.nlm.nih.gov/system/files/pub9752.pdf)
3. [NIH - Malaria Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)
4. [Carlos Atico Azira’s excellent write up](https://blog.insightdatascience.com/https-blog-insightdatascience-com-malaria-hero-a47d3d5fc4bb)
5. [Zeit Production from fast.ai](https://github.com/fastai/course-v3/tree/master/docs/production)

## LICENSE

The code in this repository is distributed under the MIT License. <br/>
Feel free to fork and try it on your own!
