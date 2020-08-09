## Project: Data Lake

1. Purpose and summary of the project
Sparkify is startup that runs a new music streaming app. Their data has grown to a point that they want to move their data warehouse to a data lake. The raw data on user activity on the app and metadata on songs resides in S3 buckets. The AWS data lake is to provide data in way that enables data analysts.

2. State and justify your database schema design and ETL pipeline
As described in the Udacity Data Modeling course, the star schema is used to model the data since data analyses on song listening data is the goal. The star schema simplifies queries.
Table songlpays is the fact table while user, songs, artists, time and users are the dimension tables (information to answer business questions). 

**Table songplays:**
Records in log data associated with song plays i.e. records with page NextSong
Contains user level the person had when playing the song
Fact table

**Table users:**
Users in the app
Since the focus is data analytics on songplays, up and downgrading of users is not handled
Dimension table

**Table songs:**
Songs in music database
Dimension table

**Table time:**
Timestamps of records in songplays broken down into specific units
Dimension table

**Table artists:**
Artists in music database
Dimension table

3. Files
**Song dataset (as per project specifications)**
Song data is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID

**Log dataset (as per project specifications)**
Log files are in JSON format generated by event simulator based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations.
The log files in the dataset you'll be working with are partitioned by year and month.

**etl.py**
Load data in JSON files from S3, process and save in parquet files on S3 or workspace 
![Visualization of ETL](/schema.png)

4. Specify cluster and run python scripts
a. Populate dl.cfg with AWS access key id and AWS secret access key
b. Create an S3 bucket and the Elastic MapReduce (EMR) cluster. Ensure that user can connect to the cluster
c. Populate output_data in main()
d. Run etl.py on EMR cluster or alternatively, run in workspace open Jupyter