#Data preparation

Before starting the data preprocessing the data should be divided into train/val/test directories (it is important that they have these 3 exact names). Each directory should have one rollout inside with corresponding images and xml files:

    ├── train
    │   ├── rollout_a1_dir           
    │   ├── rollout_a2_dir                            
    │   └── ...                 
    ├── val
    │   ├── rollout_b1_dir           
    │   ├── rollout_b2_dir           
    │   └── ...                 
    ├── test
    │   ├── rollout_c1_dir           
    │   ├── rollout_c2_dir           
    │   └── ...                 

All data preprocessing scripts use the same params file. Edit the [params/data_preprocessing.yaml](params/data_preprocessing.yaml) according to your data.
The final dataset for the models should be of the type npz. 
To convert the images and XML files into npz for you should perform the next steps:

### Step 1 -  Resize the raw images
First resize the raw images to desired output size. The one used in the scope of this project was 64x64 pixels. You can make other param files or if you change this one, you can omit it in the command because it is the default params file. 
```bash
python data_preprocessing/img_resizer.py --params params/data_preprocessing.yaml
```

### Step 2 - Match images and position
In this step each image is matched with a position. The position is extracted from the .xml files. The result is a .json file for each rollout containing the image path and the position array. 

```bash
python data_preprocessing/data_generator.py --params params/data_preprocessing.yaml
```

### Step 3 - Data interpolation and .npz generation

The script `simple_data_interpolation.py` generates .npz files used for training. One rollout is one .npz file containing the complete sequence of observations and actions.
Before creating .npz files, we have to preprocess the data. In the params file we first set `plot_hist=True` and `save_data=False`. 
By plotting the histogram we want to determine the variance of the data and decide what are outliers. Outliers are then filtered py deciding the limit variable.

<b> Outlier removal</b> 

The dataset contains outliers which are visible in high differences between two consequent positions. To asses what outliers will be removed, by setting the parameter `plot_histogram: True` and `save_data: False`, histograms of all possible position values will be plotted. The actual data has small variance so it is easy to determine how to set the  `limit` paramater, which desides the border between outliers and data that will be used. 


After setting the `limit`, change the parameters to `plot_histogram: False` and `save_data: True` and run the script again. The rollouts with outliers will be broken at the outlier point into two smaller rollouts etc.
After setting the `limit`, change the parameters to `plot_histogram: False` and `save_data: True` and run the script again. The rollouts with outliers will be broken at the outlier point into two smaller rollouts etc.


```bash
python data_preprocessing/simple_data_interpolation.py --params params/data_preprocessing.yaml
```
