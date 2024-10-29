# visual-dtw

An implementation of dynamic time warping using visual features for visual speech recognition

### Quickstart: 

#### Server:
```
# to run the development server 
make start ENV=dev

# check setup ok
curl http://127.0.0.1:5000/pava/api/v1/about

# goto http://127.0.0.1:5000/pava/api/v1 for more info
```

<!--#### R&D:-->
<!--- Setup: -->
<!--```-->
<!--# create virtual environment-->
<!--apt-get install virtualenv-->
<!--virtualenv --python=python3 /path/to/env-->
<!--source /path/to/env/bin/activate-->

<!--# install requirements-->
<!--cd visual-dtw && pip install -r requirements.txt-->
<!--export PYTHONPATH=app:$PYTHONPATH-->
<!--export DATABASE_HOST='0.0.0.0'-->
<!--```-->
<!--- Experimentation - e.g. to show the CMC curve and confusion matrix for reference users {1, 2, 3, 4, 5, 6, 7, 18} and reference sessions-->
<!--{1, 2, 3} vs test users {3} and test sessions {4}, with the top 22 predictions and by increasing # of users:-->
<!--```-->
<!--python app/main/research/experiment.py 1-7,18 1-3 3 4 --increasing_by='users'-->

<!--# for help see-->
<!--python app/main/research/experiment.py --help-->
<!--```-->
<!--- Cross-validation - e.g. to show 5-fold cross-validation for all users based on their individual sessions-->
<!--```-->
<!--python app/main/research/cross_validation.py all --num_folds=5-->

<!--# for help see-->
<!--python app/main/research/cross_validation.py --help-->
<!--```-->

#### Test suite:
Unit and integration tests are ran within a docker container.
The flake8 library is used to check for PEP-8 & PEP-257 standards. 
A coverage report is also generated to show parts of the codebase that are missing in the tests.
```
# to run the tests
make start ENV=test

# clean a docker environment (removes containers, images, networks & volumes)
make clean ENV=test
make clean ENV=dev

# or clean all environments
make clean_all

# load tests
locust -f app/tests/load/locust.py
```

#### Code Profiling: 
To profile lines of code, we use the python library "line_profiler". All you need to do is add the @profile decorator to a function and run with kernprof.
The following is an example: 
```
# script.py
@profile
def hello_world(): 
    print('Hello') 
    time.sleep(5)
    print('World')
    
if __name__ == '__main__': 
    hello_world()

# terminal
kernprof -v -l script.py
```
This will give you a line-by-line breakdown of the time it takes to complete each line and the % contribution of time for each line of the function. 

#### TODO: 
- Add type checkers?
- Use docker container python library
- Add better validation for fields: See https://aviaryan.com/blog/gsoc/restplus-validation-custom-fields
    - Need to make them consistent with rest of API json responses