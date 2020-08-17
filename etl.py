import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour,\
    weekofyear, date_format, monotonically_increasing_id
from pyspark.sql.types import StructType as R, StructField as Fld,\
    DoubleType as Dbl, StringType as Str, IntegerType as Int,\
    TimestampType as Timestamp, DateType as Date, LongType as Long


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_KEY']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_KEY']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    This procedure creates a Spark session to process data.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data_songs, output_data):
    """
    This procedure loads data from S3, processes it using Spark and then writes data back to S3.
      
    INPUTS:
    * spark - the Spark session
    * input_data_songs - location of the input data of JSON song files
    * output_data - location of where the output data will be stored
    
     OUTPUTS:
    * Songs table - output folder with parquet files of Song data
    * Artists table -  output folder with parquet files of Artist data     
    """
    # get filepath to song data file
    song_data = input_data_songs        
    
    # read song data file
    song_data_schema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Str()),
        Fld("artist_longitude", Str()),
        Fld("artist_location", Str()),
        Fld("artist_name", Str()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("duration", Dbl()),
        Fld("year", Int())
    ])
    
    df = spark.read.json(song_data,schema=song_data_schema)
    
    # extract columns to create songs table
    df.createOrReplaceTempView("songs_table")

    songs_table = spark.sql("""
        SELECT DISTINCT
            song_id, 
            title, 
            artist_id, 
            year, 
            duration
        FROM 
           songs_table
    """) 
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(
        output_data + "songs_table.parquet",
        mode="overwrite",
        partitionBy=["year", "artist_id"]
    ) 

    # extract columns to create artists table 
    df.createOrReplaceTempView("artists_table")

    artists_table = spark.sql("""
        SELECT DISTINCT
            artist_id, 
            artist_name as name, 
            artist_location as location, 
            artist_latitude as latitude, 
            artist_longitude as longitude
        FROM 
           artists_table
    """) 
    
    # write artists table to parquet files 
    artists_table.write.parquet(
        output_data + "artists_table.parquet",
        mode="overwrite"
    ) 


def process_log_data(spark, input_data_songs, input_data_logs, output_data):
   """
    This procedure loads data from S3, processes it using Spark and then writes data back to S3.
      
    INPUTS:
    * spark - the Spark session
    * input_data_songs - location of the input data of JSON song files
    * input_data_logs - location of the input data of JSON log files
    * output_data - location of where the output data will be stored
    
     OUTPUTS:
    * Users table - output folder with parquet files of Users data
    * Time table -  output folder with parquet files of Time data  
    * Songplays table -  output folder with parquet files of Songplays data     
    """
    # get filepath to log data file
    log_data = input_data_logs
     
    # read log data file
    log_data_schema = R([
        Fld("artist", Str()),
        Fld("auth", Str()),
        Fld("firstName", Str()),
        Fld("gender", Str()),
        Fld("itemInSession", Int()),
        Fld("lastName", Str()),
        Fld("length", Dbl()),
        Fld("level", Str()),
        Fld("location", Str()),
        Fld("method", Str()),
        Fld("page", Str()),
        Fld("registration", Str()),
        Fld("sessionId", Int()),
        Fld("song", Str()),
        Fld("status", Str()),
        Fld("ts", Long()),
        Fld("userAgent", Str()),
        Fld("userId", Int())
    ])
    
    df = spark.read.json(log_data,schema=log_data_schema)
    
    # filter by actions for song plays
    df_filtered = df.filter(df.page == 'NextSong')
    
    # extract columns for users table    
    df_filtered.createOrReplaceTempView("users_table")

    users_table = spark.sql("""
        SELECT DISTINCT
            userId as user_id,
            firstName as first_name,
            lastName as last_name, 
            gender,
            level
        FROM 
           users_table
    """)  
    
    # write users table to parquet files
    users_table.write.parquet(                                    
        output_data + "users_table.parquet", mode="overwrite"
    )                                                             

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000.0),\
                        Timestamp())
    df_filtered = df_filtered.withColumn('timestamp', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x / 1000)\
                       .strftime('%Y-%m-%d %H:%M:%S'), Str())
    df_filtered = df_filtered.withColumn("datetime", get_datetime("ts"))
        
    # extract columns to create time table 
    df_filtered.createOrReplaceTempView("time_table")

    time_table = spark.sql("""
        SELECT DISTINCT
            datetime as start_time,
            hour(timestamp) as hour,
            day(timestamp) as day, 
            weekofyear(timestamp) as week,
            month(timestamp) as month,
            year(timestamp) as year,
            dayofweek(timestamp) as weekday
        FROM 
           time_table
    """)  
    
    # write time table to parquet files partitioned by year and month 
    time_table.write.parquet(
        output_data + "time_table.parquet",
        mode="overwrite",
        partitionBy=["year", "month"]
    ) 

    # read in song data to use for songplays table
    song_data = input_data_songs
    
     # read song data file
    song_data_schema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Str()),
        Fld("artist_longitude", Str()),
        Fld("artist_location", Str()),
        Fld("artist_name", Str()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("duration", Dbl()),
        Fld("year", Int())
    ])
    
    song_df = spark.read.json(song_data,schema=song_data_schema)
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_df = df_filtered.join(song_df, (df_filtered.artist == song_df.artist_name)\
        & (df_filtered.song == song_df.title))
        
    songplays_df = songplays_df.withColumn("songplay_id", monotonically_increasing_id())
    
    songplays_df.createOrReplaceTempView("songplays_table")
    
    songplays_table = spark.sql("""
        SELECT  DISTINCT
            songplay_id,
            timestamp as start_time,
            userId as user_id,
            level,
            song_id,
            artist_id,
            sessionId as session_id,
            location,
            userAgent as user_agent,
            year(timestamp) as year,
            month(timestamp) as month
        FROM 
            songplays_table 
    """)

    # write songplays table to parquet files partitioned by year and month 
    songplays_table.write.parquet(
        output_data + "songplays_table.parquet",
        mode="overwrite",
        partitionBy=["year", "month"]
    ) 
    
    
def main():
    spark = create_spark_session()
    
    input_data_songs = config['S3']['SONG_DATA']
    input_data_logs = config['S3']['LOG_DATA']
    output_data = config['S3']['OUTPUT_DATA']
    
    process_song_data(spark, input_data_songs, output_data)    
    process_log_data(spark, input_data_songs, input_data_logs, output_data)


if __name__ == "__main__":
    main()