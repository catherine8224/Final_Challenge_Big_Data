from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import datetime, timedelta
import json
import numpy as np
import sys

def main(sc, spark):
  '''
  Transfer our code from the notebook here, however, remember to replace
  the file paths with the ones provided in the problem description.
  '''
  dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
  dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
  #dfPlaces = spark.read.csv('core-places-nyc.csv', header=True, escape='"')
  #dfPattern = spark.read.csv('weekly-patterns-nyc-2019-2020-sample.csv', header=True, escape='"')
  OUTPUT_PREFIX = sys.argv[1]
  CAT_CODES = {'445210', '722515', '445299', '445120', '452210', '311811', '722410', '722511', '445220', '445292', '445110', '445291', '445230', '446191', '446110', '722513', '452311'}
  CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446191': 5, '446110': 5, '722515': 6, '311811': 6, '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7, '445210': 7, '445110': 8}
  dfD = dfPlaces.filter(dfPlaces.naics_code.isin(CAT_CODES)).select("placekey", "naics_code")
  udfToGroup = F.udf(lambda x: CAT_GROUP[x])

  dfE = dfD.withColumn('group', udfToGroup('naics_code'))
  dfF = dfE.drop('naics_code').cache()
  groupCount = dict(dfF.groupBy("group").count().collect())

  def expandVisits(date_range_start, visits_by_day):
      visits_by_day = json.loads(visits_by_day)
      for i in range(len(visits_by_day)):
        overall_date = (datetime.strptime(date_range_start[:10], "%Y-%m-%d")+ timedelta(days=i))
        if overall_date.year != 2018:
          yield overall_date.year, overall_date.strftime('%Y-%m-%d')[5:10], visits_by_day[i]

  visitType = T.StructType([T.StructField('year', T.IntegerType()),
                            T.StructField('date', T.StringType()),
                            T.StructField('visits', T.IntegerType())])

  udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

  dfH = dfPattern.join(dfF, 'placekey') \
      .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
      .select('group', 'expanded.*')

  def computeStats(group, visits):
    add_zeros = groupCount[group] - len(visits)
    visits.extend([0]*add_zeros)
    median = np.median(visits) #get the median of values in a list in each row
    if median - np.std(visits) >=0:
      return (int(median), int(median - np.std(visits)), int(median + np.std(visits)))
    else:
      return (int(median), 0, int(median + np.std(visits)))


  statsType = T.StructType([T.StructField('median', T.IntegerType()),
                            T.StructField('low', T.IntegerType()),
                            T.StructField('high', T.IntegerType())])

  udfComputeStats = F.udf(computeStats, statsType)

  dfI = dfH.groupBy('group', 'year', 'date') \
      .agg(F.collect_list('visits').alias('visits')) \
      .withColumn('stats', udfComputeStats('group', 'visits'))

  dfJ = dfI \
      .select('group', 'year', F.concat(F.lit("2020-"), F.col('date')).alias('date'), 'stats.*')\
      .sort('group', 'year', 'date') \
      .cache()

  group_names = {'big_box_grocers': 0,
  'convenience_stores': 1,
  'drinking_places': 2,
  'full_service_restaurants': 3,
  'limited_service_restaurants': 4,
  'pharmacies_and_drug_stores': 5,
  'snack_and_retail_bakeries': 6,
  'specialty_food_stores': 7,
  'supermarkets_except_convenience_stores': 8}

  for filename,number in group_names.items():
    dfJ.filter('group='+str(number)) \
        .drop('group').coalesce(1).write.csv(OUTPUT_PREFIX+'/'+filename,
                  mode='overwrite', header=True)

  #!ls /content/ | grep -Ev ".csv|sample_data"

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)
