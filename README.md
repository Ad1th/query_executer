To run on your local

1. Start a pipenv shell with "pipenv shell"
2. cd query_executor/app
3. export PYTHONPATH=$(dirname "$PWD")
4. uvicorn main:app --workers 1 --log-level info --port 8082

This starts app on localhost:8082.

Create a copy of config/main/settings.yaml to config/local/settings.yaml and use config values for your own local


pipenv run uvicorn main:app --workers 1 --log-level info --port 8082