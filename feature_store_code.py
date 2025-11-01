import sagemaker
import pandas as pd
import io
from time import gmtime, strftime, sleep
import datetime
import random
import time
from sagemaker.feature_store.feature_group import FeatureGroup

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()

def random_datetime(yr: int) -> float:
    delta = datetime.timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return (datetime.datetime(1900 + int(yr), 1, 1) + delta).timestamp()

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
col_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']

df = pd.read_csv(data_url, sep=r'\s+', header=None, names=col_names, na_values='?')

df["car_name"] = df["car_name"].astype('string')
df["release_date"] = df["model_year"].apply(random_datetime)
df["event_time"] = df["release_date"].apply(lambda ts: (datetime.datetime.fromtimestamp(ts) + datetime.timedelta(days=365)).timestamp())
df["metadata_0"] = pd.Series(["empty"] * len(df), dtype="string")
df["metadata_1"] = pd.Series(["empty"] * len(df), dtype="string")
df["metadata_2"] = pd.Series(["empty"] * len(df), dtype="string")
df["metadata_3"] = pd.Series(["empty"] * len(df), dtype="string")

df.info()


def check_feature_group_status(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    print(f'Current status: {status}')
    while status == "Creating":
        print("Waiting for Feature Group to be Created")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    print(f"FeatureGroup {feature_group.name} successfully created.")


feature_group_name = 'poc-0'
print(f'Feature group name: {feature_group_name}')

feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sess)
feature_group.load_feature_definitions(data_frame=df)
feature_group.create(
    s3_uri=f's3://{bucket}/{feature_group_name}',  # s3 bucket location for the offline store
    enable_online_store=True,  # Create online store for this feature group
    record_identifier_name='car_name',  # record identifier
    event_time_feature_name='event_time',  # event_time feature required by the feature store
    description='This feature group tracks the vehicle information such as mpg, and horsepower between 1970 and 1982.',
    role_arn=role
)

check_feature_group_status(feature_group)
feature_group.describe()

featurestore_runtime = sess.boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)


def find_record(record_id):
    online_record = featurestore_runtime.get_record(FeatureGroupName=feature_group.name, RecordIdentifierValueAsString=record_id)
    print(f"Online record: {online_record}")

    query = feature_group.athena_query()
    table_name = query.table_name
    print(table_name)

    sql_query = f"""
    SELECT *
    FROM "{table_name}"
    where car_name = '{record_id}'
    """
    query.run(query_string=sql_query,output_location=f's3://{bucket}/queries/{feature_group_name}/query_results/')
    query.wait()
    offline_record = query.as_dataframe()
    print(f'Offline record: {offline_record}')


find_record('amc ambassador dpl') # record is only returned from the online store
find_record('amc concord') # record is returned from both the online and offline store