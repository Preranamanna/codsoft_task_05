import tensor flow as tf
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
from tensorflow.keras.applications
import VGG16
from tensorflow.keras.layersimportDenseFlattenDropout
from tensorflow.keras.models
import Model
import numpy as np
import pandas as pd
import matplotlib.py plot as plt
import zip file
import os

#Define the directory where you want to save the data set
dataset_dir='/content/drive/MyDrive/food_101'

#Ensure the directory exists create it if necessary
os.makedirs(dataset_direxist_ok=True)

#Change to the dataset directory
os.chdir(dataset_dir)

#Download the data 
print(\Downloadingthedata...\)
!wget-P{dataset_dir}http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
print(\Datasetdownloaded!\)

#Extract the data
print(Extractingdata..)
!tarxzvffood-101.tar.gz>/dev/null2>&1
print(Extractiondone!)

Downloading the data...
--2024-06-2006:00:45--http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
Resolvingdata.vision.ee.ethz.ch(data.vision.ee.ethz.ch)...129.132.52.1782001:67c:10ec:36c2::178
Connectingtodata.vision.ee.ethz.ch(data.vision.ee.ethz.ch)|129.132.52.178|:80...connected.
HTTPrequestsentawaitingresponse...302Found
Location:https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz[following]
--2024-06-2006:00:46--https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
Connectingtodata.vision.ee.ethz.ch(data.vision.ee.ethz.ch)|129.132.52.178|:443...connected.
HTTPrequestsentawaitingresponse...^C
Datasetdownloaded!
Extractingdata..
^C
Extractiondone!

pipinstalltensorflownumpypandasmatplotlibtqdm

Requirementalreadysatisfied:tensorflowin/usr/local/lib/python3.10/dist-packages(2.15.0)
Requirementalreadysatisfied:numpyin/usr/local/lib/python3.10/dist-packages(1.25.2)
Requirementalreadysatisfied:pandasin/usr/local/lib/python3.10/dist-packages(2.0.3)
Requirementalreadysatisfied:matplotlibin/usr/local/lib/python3.10/dist-packages(3.7.1)
Requirementalreadysatisfied:tqdmin/usr/local/lib/python3.10/dist-packages(4.66.4)
Requirementalreadysatisfied:absl-py>=1.0.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(1.4.0)
Requirementalreadysatisfied:astunparse>=1.6.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(1.6.3)
Requirementalreadysatisfied:flatbuffers>=23.5.26in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(24.3.25)
Requirementalreadysatisfied:gast!=0.5.0!=0.5.1!=0.5.2>=0.2.1in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(0.5.4)
Requirementalreadysatisfied:google-pasta>=0.1.1in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(0.2.0)
Requirementalreadysatisfied:h5py>=2.9.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(3.9.0)
Requirementalreadysatisfied:libclang>=13.0.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(18.1.1)
Requirementalreadysatisfied:ml-dtypes~=0.2.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(0.2.0)
Requirementalreadysatisfied:opt-einsum>=2.3.2in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(3.3.0)
Requirementalreadysatisfied:packagingin/usr/local/lib/python3.10/dist-packages(fromtensorflow)(24.1)
Requirementalreadysatisfied:protobuf!=4.21.0!=4.21.1!=4.21.2!=4.21.3!=4.21.4!=4.21.5<5.0.0dev>=3.20.3in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(3.20.3)
Requirementalreadysatisfied:setuptoolsin/usr/local/lib/python3.10/dist-packages(fromtensorflow)(67.7.2)
Requirementalreadysatisfied:six>=1.12.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(1.16.0)
Requirementalreadysatisfied:termcolor>=1.1.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(2.4.0)
Requirementalreadysatisfied:typing-extensions>=3.6.6in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(4.12.2)
Requirementalreadysatisfied:wrapt<1.15>=1.11.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(1.14.1)
Requirementalreadysatisfied:tensorflow-io-gcs-filesystem>=0.23.1in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(0.37.0)
Requirementalreadysatisfied:grpcio<2.0>=1.24.3in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(1.64.1)
Requirementalreadysatisfied:tensorboard<2.16>=2.15in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(2.15.2)
Requirementalreadysatisfied:tensorflow-estimator<2.16>=2.15.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(2.15.0)
Requirementalreadysatisfied:keras<2.16>=2.15.0in/usr/local/lib/python3.10/dist-packages(fromtensorflow)(2.15.0)
Requirementalreadysatisfied:python-dateutil>=2.8.2in/usr/local/lib/python3.10/dist-packages(frompandas)(2.8.2)
Requirementalreadysatisfied:pytz>=2020.1in/usr/local/lib/python3.10/dist-packages(frompandas)(2023.4)
Requirementalreadysatisfied:tzdata>=2022.1in/usr/local/lib/python3.10/dist-packages(frompandas)(2024.1)
Requirementalreadysatisfied:contourpy>=1.0.1in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(1.2.1)
Requirementalreadysatisfied:cycler>=0.10in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(0.12.1)
Requirementalreadysatisfied:fonttools>=4.22.0in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(4.53.0)
Requirementalreadysatisfied:kiwisolver>=1.0.1in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(1.4.5)
Requirementalreadysatisfied:pillow>=6.2.0in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(9.4.0)
Requirementalreadysatisfied:pyparsing>=2.3.1in/usr/local/lib/python3.10/dist-packages(frommatplotlib)(3.1.2)
Requirementalreadysatisfied:wheel<1.0>=0.23.0in/usr/local/lib/python3.10/dist-packages(fromastunparse>=1.6.0->tensorflow)(0.43.0)
Requirementalreadysatisfied:google-auth<3>=1.6.3in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(2.27.0)
Requirementalreadysatisfied:google-auth-oauthlib<2>=0.5in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(1.2.0)
Requirementalreadysatisfied:markdown>=2.6.8in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(3.6)
Requirementalreadysatisfied:requests<3>=2.21.0in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(2.31.0)
Requirementalreadysatisfied:tensorboard-data-server<0.8.0>=0.7.0in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(0.7.2)
Requirementalreadysatisfied:werkzeug>=1.0.1in/usr/local/lib/python3.10/dist-packages(fromtensorboard<2.16>=2.15->tensorflow)(3.0.3)
Requirementalreadysatisfied:cachetools<6.0>=2.0.0in/usr/local/lib/python3.10/dist-packages(fromgoogle-auth<3>=1.6.3->tensorboard<2.16>=2.15->tensorflow)(5.3.3)
Requirementalreadysatisfied:pyasn1-modules>=0.2.1in/usr/local/lib/python3.10/dist-packages(fromgoogle-auth<3>=1.6.3->tensorboard<2.16>=2.15->tensorflow)(0.4.0)
Requirementalreadysatisfied:rsa<5>=3.1.4in/usr/local/lib/python3.10/dist-packages(fromgoogle-auth<3>=1.6.3->tensorboard<2.16>=2.15->tensorflow)(4.9)
Requirementalreadysatisfied:requests-oauthlib>=0.7.0in/usr/local/lib/python3.10/dist-packages(fromgoogle-auth-oauthlib<2>=0.5->tensorboard<2.16>=2.15->tensorflow)(1.3.1)
Requirementalreadysatisfied:charset-normalizer<4>=2in/usr/local/lib/python3.10/dist-packages(fromrequests<3>=2.21.0->tensorboard<2.16>=2.15->tensorflow)(3.3.2)
Requirementalreadysatisfied:idna<4>=2.5in/usr/local/lib/python3.10/dist-packages(fromrequests<3>=2.21.0->tensorboard<2.16>=2.15->tensorflow)(3.7)
Requirementalreadysatisfied:urllib3<3>=1.21.1in/usr/local/lib/python3.10/dist-packages(fromrequests<3>=2.21.0->tensorboard<2.16>=2.15->tensorflow)(2.0.7)
Requirementalreadysatisfied:certifi>=2017.4.17in/usr/local/lib/python3.10/dist-packages(fromrequests<3>=2.21.0->tensorboard<2.16>=2.15->tensorflow)(2024.6.2)
Requirementalreadysatisfied:MarkupSafe>=2.1.1in/usr/local/lib/python3.10/dist-packages(fromwerkzeug>=1.0.1->tensorboard<2.16>=2.15->tensorflow)(2.1.5)
Requirementalreadysatisfied:pyasn1<0.7.0>=0.4.6in/usr/local/lib/python3.10/dist-packages(frompyasn1-modules>=0.2.1->google-auth<3>=1.6.3->tensorboard<2.16>=2.15->tensorflow)(0.6.0)
Requirementalreadysatisfied:oauthlib>=3.0.0in/usr/local/lib/python3.10/dist-packages(fromrequests-oauthlib>=0.7.0->google-auth-oauthlib<2>=0.5->tensorboard<2.16>=2.15->tensorflow)(3.2.2)

fromgoogle.colabimportdrive
drive.mount('/content/drive')

Drivealreadymountedat/content/drive;toattempttoforciblyremountcalldrive.mount(\/content/drive\force_remount=True).

extract_path='/content/drive/MyDrive/food_101/food-101'

#Assuming the images are extracted to'images'folder within'Food-101'
image_dir=os.path.join(extract_path'images')

#Function to load images and labels

from tqdm import tqdm
import cv2

defload_images_from_directory(directoryimg_size=(6464)):
images=[]
labels=[]
classes=os.listdir(directory)
forclass_nameintqdm(classesdesc=\Loadingimages\):
class_dir=os.path.join(directoryclass_name)
ifos.path.isdir(class_dir):
forimg_fileinos.listdir(class_dir):
img_path=os.path.join(class_dirimg_file)
try:
img=cv2.imread(img_path)
ifimgisnotNone:
img=cv2.resize(imgimg_size)
images.append(img)
labels.append(class_name)
exceptExceptionase:
print(f\Errorloadingimage{img_path}:{e}\)
returnnp.array(images)np.array(labels)

#Load the data
Xy=load_images_from_directory(image_dir)

Loading images:100%|██████████|101/101[1:18:33<00:0046.66s/it]

print(f\Numberofimagesloaded:{len(X)}\)

Number of images loaded:101000

#Encode labels to integers

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)

#split the data into train and test
from sklearn.model_selectionimporttrain_test_split
X_trainX_testy_trainy_test=train_test_split(Xy_encodedtest_size=0.2random_state=42)

#Convert labels to one-hot encoding
fromtensorflow.keras.utilsimportto_categorical
num_classes=len(np.unique(y_encoded))
y_train=to_categorical(y_trainnum_classes=num_classes)
y_test=to_categorical(y_testnum_classes=num_classes)

definingthemodel

defcreate_cnn_model(input_shapenum_classes):
model=Sequential([
Conv2D(32(33)activation='relu'input_shape=input_shape)
MaxPooling2D((22))
Conv2D(64(33)activation='relu')
MaxPooling2D((22))
Conv2D(128(33)activation='relu')
MaxPooling2D((22))
Flatten()
Dense(256activation='relu')
Dropout(0.5)
Dense(num_classesactivation='softmax')
])
model.compile(optimizer='adam'loss='categorical_crossentropy'metrics=['accuracy'])
returnmodel


#Create the model
fromtensorflow.keras.modelsimportSequential
fromtensorflow.keras.layersimportConv2DMaxPooling2D
input_shape=X_train.shape[1:]#(64643)
model=create_cnn_model(input_shapenum_classes)

#Train the model
history=model.fit(X_trainy_trainepochs=100validation_split=0.2batch_size=32)
model.save('food_recognition_model.task5')

Epoch1/100
463/2020[=====>........................]-ETA:6:30-loss:5.3022-accuracy:0.0104

last_epoch_loss=history.history['loss'][-1]
last_epoch_accuracy=history.history['accuracy'][-1]

print(f\Lastepochloss:{last_epoch_loss:.4f}\)
print(f\Lastepochaccuracy:{last_epoch_accuracy:.4f}\)

#Assuming we have a dictionary of calories per food item
calorie_data={'apple_pie':0
'baby_back_ribs':1
'baklava':2
'beef_carpaccio':3
'beef_tartare':4
'beet_salad':5
'beignets':6
'bibimbap':7
'bread_pudding':8
'breakfast_burrito':9
'bruschetta':10
'caesar_salad':11
'cannoli':12
'caprese_salad':13
'carrot_cake':14
'ceviche':15
'cheese_plate':16
'cheesecake':17
'chicken_curry':18
'chicken_quesadilla':19
'chicken_wings':20
'chocolate_cake':21
'chocolate_mousse':22
'churros':23
'clam_chowder':24
'club_sandwich':25
'crab_cakes':26
'creme_brulee':27
'croque_madame':28
'cup_cakes':29
'deviled_eggs':30
'donuts':31
'dumplings':32
'edamame':33
'eggs_benedict':34
'escargots':35
'falafel':36
'filet_mignon':37
'fish_and_chips':38
'foie_gras':39
'french_fries':40
'french_onion_soup':41
'french_toast':42
'fried_calamari':43
'fried_rice':44
'frozen_yogurt':45
'garlic_bread':46
'gnocchi':47
'greek_salad':48
'grilled_cheese_sandwich':49
'grilled_salmon':50
'guacamole':51
'gyoza':52
'hamburger':53
'hot_and_sour_soup':54
'hot_dog':55
'huevos_rancheros':56
'hummus':57
'ice_cream':58
'lasagna':59
'lobster_bisque':60
'lobster_roll_sandwich':61
'macaroni_and_cheese':62
'macarons':63
'miso_soup':64
'mussels':65
'nachos':66
'omelette':67
'onion_rings':68
'oysters':69
'pad_thai':70
'paella':71
'pancakes':72
'panna_cotta':73
'peking_duck':74
'pho':75
'pizza':76
'pork_chop':77
'poutine':78
'prime_rib':79
'pulled_pork_sandwich':80
'ramen':81
'ravioli':82
'red_velvet_cake':83
'risotto':84
'samosa':85
'sashimi':86
'scallops':87
'seaweed_salad':88
'shrimp_and_grits':89
'spaghetti_bolognese':90
'spaghetti_carbonara':91
'spring_rolls':92
'steak':93
'strawberry_shortcake':94
'sushi':95
'tacos':96
'takoyaki':97
'tiramisu':98
'tuna_tartare':99
'waffles':100}


#Map labels to calories
y_calories=np.array([calorie_data[label]forlabeliny])

#Split the data for calorie prediction
X_train_calX_test_caly_train_caly_test_cal=train_test_split(Xy_caloriestest_size=0.2random_state=42)

#regression model for calorie estimation
defcreate_regression_model(input_shape):
model=Sequential([
Conv2D(32(33)activation='relu'input_shape=input_shape)
MaxPooling2D((22))
Conv2D(64(33)activation='relu')
MaxPooling2D((22))
Flatten()
Dense(256activation='relu')
Dropout(0.5)
Dense(1)#Outputlayerforregression
])
model.compile(optimizer='adam'loss='mean_squared_error'metrics=['mae'])
return model

#Create the calorie estimation model
calorie_model=create_regression_model(input_shape)

#Train the model
calorie_model.fit(X_train_caly_train_calepochs=15validation_split=0.2batch_size=32)

#Save the trained model
calorie_model.save('calorie_model.task5')

#Evaluate the model
msemae=calorie_model.evaluate(X_test_caly_test_cal)
print(f'TestMAEforcalorieestimation:{mae}')

Epoch1/15
2020/2020[==============================]-538s266ms/step-loss:2137.5095-mae:28.6993-val_loss:879.2755-val_mae:25.2536
Epoch2/15
2020/2020[==============================]-515s255ms/step-loss:1067.5028-mae:27.1581-val_loss:920.3135-val_mae:25.7035
Epoch3/15
2020/2020[==============================]-511s253ms/step-loss:1047.7612-mae:26.8930-val_loss:883.1783-val_mae:25.1975
Epoch4/15
2020/2020[==============================]-493s244ms/step-loss:1023.7082-mae:26.5670-val_loss:893.3580-val_mae:25.3492
Epoch5/15
2020/2020[==============================]-511s253ms/step-loss:1007.3010-mae:26.3639-val_loss:905.7197-val_mae:25.3884
Epoch6/15
2020/2020[==============================]-522s258ms/step-loss:993.8100-mae:26.2173-val_loss:871.9583-val_mae:25.2176
Epoch7/15
2020/2020[==============================]-497s246ms/step-loss:966.7722-mae:25.8583-val_loss:907.4570-val_mae:25.5909
Epoch8/15
2020/2020[==============================]-498s247ms/step-loss:941.8436-mae:25.4989-val_loss:900.7116-val_mae:25.3235
Epoch9/15
2020/2020[==============================]-450s223ms/step-loss:913.5693-mae:25.1227-val_loss:898.5187-val_mae:25.3479
Epoch10/15
2020/2020[==============================]-455s225ms/step-loss:895.1827-mae:24.8336-val_loss:947.9779-val_mae:25.8960
Epoch11/15
2020/2020[==============================]-465s230ms/step-loss:868.9941-mae:24.4334-val_loss:961.4987-val_mae:25.9833
Epoch12/15
2020/2020[==============================]-468s231ms/step-loss:840.6421-mae:24.0346-val_loss:910.7772-val_mae:25.4432
Epoch13/15
2020/2020[==============================]-448s222ms/step-loss:828.6833-mae:23.8449-val_loss:913.5120-val_mae:25.4797
Epoch14/15
2020/2020[==============================]-463s229ms/step-loss:806.6546-mae:23.5021-val_loss:915.7200-val_mae:25.4938
Epoch15/15
2020/2020[==============================]-450s223ms/step-loss:785.0123-mae:23.1543-val_loss:918.5875-val_mae:25.4425
632/632[==============================]-38s60ms/step-loss:924.7657-mae:25.5113
TestMAEforcalorieestimation:25.511268615722656

fromsklearn.metricsimportconfusion_matrix

#Predictions on test data
y_pred_cal=calorie_model.predict(X_test_cal)
print(y_pred_cal)
print(y_test_cal)
fromsklearn.metricsimportmean_absolute_errormean_squared_error

mae=mean_absolute_error(y_test_caly_pred_cal)
mse=mean_squared_error(y_test_caly_pred_cal)
print(f\MAE:{mae:.2f}MSE:{mse:.2f}\)

632/632[==============================]-41s65ms/step
[[55.207855]
[70.321976]
[28.216515]
...
[46.66958]
[57.847107]
[55.663723]]
[4775100...21243]
MAE:25.51MSE:924.77

VisualizePredictions

importmatplotlib.pyplotasplt


#Create a scatter plot
plt.scatter(y_test_caly_pred_calalpha=0.2)
plt.xlabel(\ActualCalories\)
plt.ylabel(\PredictedCalories\)
plt.title(\Actualvs.PredictedCalories\)
plt.grid(True)
plt.show()
