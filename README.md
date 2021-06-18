#Earthquake
1. From time series data, for each segment 150000 long i compute about 100 features (features_extraction_module.py)

Train data set is about 600 000 000 lenght
Segment "i" is from (0 + i*150000: (i+1) * 150000)
My data look like this:
(98 features) (time_to_failure)

2.I get some extra data for time_to_failure higher than 9
by computing feature in this region with shift 50000

Segment "i j" is from (0 + i*150000 + j*50000: (i+1) * 150000+ j *50000)

Point is to have more constant distribution


3. I predict value for each segment i use 5 layers deep tensorflow neural network, with dropout. 
  

![image](https://user-images.githubusercontent.com/72819970/122527995-27c7cf80-d01c-11eb-9e5a-a6b60f7a118a.png)
