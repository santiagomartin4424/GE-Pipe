## Automatic Generation of ML-pipelines using PonyGE2

GE-Pipe is an implementation based on the [PonyGE2](https://github.com/PonyGE/PonyGE2) library (that uses Grammatical Evolution, GE), in order to solve the Automatic Workflow Composition (AWC) problem.

In this context, GE is a type of Evolutionary Algorithm that evolves a population of ML pipelines, in order to explore the solution space and find the best performing pipeline for a given dataset.

### How to run GE-Pipe
#### 1. Clone this repository, and unpack the datasets.tar.gz file
#### 2. Create a .venv enviroment in Python to execute the implementation:
```
python -m venv .venv
source .venv/bin/activate
```
If first time, execute this command to install the dependencies required
```
pip install -r requirements.txt
```

#### 3. If you want to try other datasets.
To use datasets different than the ones presented here, you can create a folder inside datasets/processed_datasets/ with the .csv files of the train partition and the test partition. 2 files in total. They should have the following naming format: mydataset_train.csv, mydataset_test.csv (for example for the abalone dataset, abalone_train.csv, abalone_test.csv).

Then, modify the _datasets_ variable inside the awc_script.sh file, and write the name of the train partition file, of the dataset you want to test (so, it should have this format: "mydataset_train").

#### 4. Adjust the desired parameters for the evolutionary process:
Inside the awc_script.sh file, specify the grammar file (default is classpip_smart_v3.bnf), and the parameters file (default is parameters_awc.txt) to use. If you want to modify the parameters file, you can get more information about the behaviour of the parameters in the [PonyGE2 wiki](https://github.com/PonyGE/PonyGE2/wiki). Also from this file, you can modify the number of executions per dataset, and the number of cpu cores to use. Optionally, you can also modify the time budget of each execution in the header of the myponyge.py file. Default is set to 1h, 3600 seconds.
#### 5. Execute the awc_script.sh file with the following command:

```
nohup ./awc_script.sh > exit.log 2>&1 &
```

#### 6. You can see the progress with:
```
cat exit.log
```
You can close the terminal because we are using nohup. 

#### 7. If desired, you can also execute this command to terminate the script before it finishes normally:
```
killall -9 awc_script.sh python3.
```











