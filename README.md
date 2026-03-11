# Statistics
DreamMend Efficacy Study

# How to start
- clone the repository: 'git clone https://github.com/jules201/Statistics.git'
- Open a terminal (or command prompt), navigate to your project folder, and run: 'python -m venv venv'
🔹 On Windows:venv\Scripts\activate
* On linux: source venv/bin/activate



- Install requirements: 'pip install -r requirements.txt'

#Set up
The Study Data is stored in the Data folder. Dataset.csv contains the data from timepoints t0-t4 (already preprocessed).


The important data sets are: t0_control.csv,...,t4_intervention.csv
In each name is the timepoint the data is collected and the respective group

The main Script is in Scripts and is called analysis.py  
* you can run it in your terminal 'python analysis.py', but you need to be in the subfolder /Scripts
* here we get some descriptive stats like the mean nightmare frequency per group, then Linear mixed model, permutation Test and Jacknife

In psqi_analysis,
* We analyze the data from the PSQI questionnaire

In Demographics, 
* we gathered some demographic data, which is not accurate since we cant match all later participants with previous participants since there are some imagined vp_codes. (so our demographic data is just a rough guess if we can call it that)

