#pybabel extract -F babel.cfg -k _l -o messages.pot .
#pybabel update -i messages.pot -d translations -l en 
#pybabel compile -f -d translations

docker run -d -p 6379:6379 redis
nohup python3.6 worker.py &
nohup python3.6 app.py &