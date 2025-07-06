#!/bin/bash

export PYTHONPATH=/home/site/wwwroot/.python_packages/lib/site-packages:$PYTHONPATH
gunicorn --bind=0.0.0.0 --timeout 600 app:app
