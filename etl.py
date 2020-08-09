import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, TimestampType as Timestamp
from pyspark.sql.types import TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS CREDS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS CREDS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    
    """
        Creates a spark session
        
            Arguments:
                None
                
            Returns:
                None
    """
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    
    """
        Loads song data from specified S3 bucket, processes data and writes them back to S3 bucket or workspace in parquet files
        
        Arguments:
            spark: Spark session
            input_data: path to song data
            output_data: path to where files are written to
        Returns:
            None
    """
    
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # create song data schema to ensure that schema is inferred correctly
    songSchema = R([
        Fld("artist_id",Str()),
        Fld("artist_latitude",Dbl()),
        Fld("artist_location",Str()),
        Fld("artist_longitude",Dbl()),
        Fld("artist_name",Str()),
        Fld("duration",Dbl()),
        Fld("num_songs",Dbl()),
        Fld("song_id",Str()),
        Fld("title",Str()),
        Fld("year",Dbl()),
    ])

    # read song data file
    df = spark.read.schema(songSchema).json(song_data)
    #df.sprintSchema()
    
    # extract columns to create songs table
    songs_table = df.select('song_id', 'title','artist_id','year','duration')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy('year','artist_id').parquet(output_data + "song.parquet")

    # extract columns to create artists table
    artists_table = df.select('artist_id', col("artist_name").alias("name"), 
                              col("artist_location").alias("location"),
                              col("artist_latitude").alias("latitude"),
                              col("artist_longitude").alias("longitude")).distinct()
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data + "artist.parquet")


def process_log_data(spark, input_data, output_data):
    
    """
        Loads log data from specified S3 bucket, processes data and writes them back to S3 bucket or workspace in parquet files
        
        Arguments:
            spark: Spark session
            input_data: path to log data
            output_data: path to where files are written to
        Returns:
            None
    """
    
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"
    
    # create log data schema to ensure that schema is inferred correctly
    logSchema = R([
        Fld("artist",Str()),
        Fld("auth",Str()),
        Fld("firstName",Str()),
        Fld("gender",Str()),
        Fld("itemInSession",Int()),
        Fld("lastName",Str()),
        Fld("length",Dbl()),
        Fld("level",Str()),
        Fld("location",Str()),
        Fld("method",Str()),
        Fld("page",Str()),
        Fld("registration",Str()),
        Fld("sessionId",Int()),
        Fld("song",Str()),
        Fld("status",Int()),
        Fld("ts",Dbl()),
        Fld("userAgent",Str()),
        Fld("userId",Str()),
    ])
    # read log data file
    df = spark.read.schema(logSchema).json(log_data)
    
    # filter by actions for song plays
    df = df[df['page']=='NextSong']

    # extract columns for users table   
    # Since the focus is data analytics on songplays, up and downgrading of users is not handled
    user_table = df.select(col('userId').alias('user_id')
                       , col('firstName').alias('first_name')
                       , col('lastName').alias('last_name')
                       , 'gender'
                       , 'level'
                      ).distinct()
    
    # write users table to parquet files
    user_table.write.mode('overwrite').parquet(output_data + "user.parquet")
    
    # convert column ts to timestamp to be able to extract hour, weekday etc.
    df = df.withColumn('ts', (F.round(col('ts')/1000)).cast(TimestampType()))
    
    # extract hour, day, week, month, year and weekday from ts
    df = df.withColumn("hour", hour(col("ts"))).withColumn("day", dayofmonth(col("ts"))).withColumn("week", weekofyear(col("ts"))).withColumn("month", month(col("ts"))).withColumn("year", year(col("ts"))).withColumn("weekday", date_format(col("ts"), "u"))
    
    # extract columns to create time table
    time_table = df.select(col('ts').alias('start_time')
                       , 'hour'
                       , 'day'
                       , 'week'
                       , 'month'
                       , 'year'
                       , 'weekday'
                      ).distinct()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy('year','month').parquet(output_data + "time.parquet")

    # read in song data to use for songplays table; no year or month to avoid trouble when joining with df
    song_df = spark.read.parquet("output/song.parquet").select('song_id', 'title','duration','artist_id')

    # extract columns from joined song and log datasets to create songplays table 
    songplay_table = df.withColumn("songplay_id", monotonically_increasing_id()).join(song_df, df.song == song_df.title, 'left_outer').select(col('ts').alias('start_time')
                                , col('userId').alias('user_id')
                                , 'level'
                                , 'song_id'
                                , 'artist_id'
                                , col('sessionId').alias('session_id')
                                , 'location'
                                , col('userAgent').alias('user_agent')
                                , 'year'
                                , 'month'
                               )

    # write songplays table to parquet files partitioned by year and month
    songplay_table.write.mode('overwrite').partitionBy('year','month').parquet(output_data + "songplay.parquet")


def main():
    
    """
        Read config data
        Create Spark session in which data in JSON files is loaded from S3 and processes that data using Spark and writes them back to S3 or workspace
        
        Arguments:
            None
        
        Returns:
            None
    """
    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "output/" #create your own bucket
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
