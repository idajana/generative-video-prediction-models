## Generating data with Pybullet

This part is used to generate synthetic data using Pybullet. The showed command will 

```bash
python gen_moving_obj.py --dir /output_path --len_roll 60 --num_roll 20
```
Where the arguments denote:

`dir  ` - output directory to store images and csv files with position

`len_roll ` - length of one rollout 

`num_roll ` - number of different rollouts to generate

This script will generate objects which are moving in different directions, it randomly chooses how many objects will it generate (maximum 3).
Next is to convert the outputs to .npz files. It is possible to specifiy resize dimension if necessary. 
```bash
python ../data_preprocessing/csv2npz.py --dir /data_root_dir
```
