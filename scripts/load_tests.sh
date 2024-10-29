host=${host:-"http://0.0.0.0:5000/"}
time=${time:-180}

locust -f app/tests/load/test.py --no-web -H ${host} -c 1 -r 1 -t ${time}s --csv 1_user
locust -f app/tests/load/test.py --no-web -H ${host} -c 5 -r 1 -t ${time}s --csv 5_user
locust -f app/tests/load/test.py --no-web -H ${host} -c 10 -r 1 -t ${time}s --csv 10_user

python3 app/tests/load/test.py 1_user_stats.csv 5_user_stats.csv 10_user_stats.csv
rm *_user_*.csv