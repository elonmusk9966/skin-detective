#!/bin/sh

echo 'export APP_PATH="/tmp/8dadf7c488f5f76"' >> ~/.bashrc
echo 'cd $APP_PATH' >> ~/.bashrc

# Enter the source directory to make sure the script runs where the user expects
cd /tmp/8dadf7c488f5f76

export APP_PATH="/tmp/8dadf7c488f5f76"
if [ -z "$HOST" ]; then
                export HOST=0.0.0.0
fi

if [ -z "$PORT" ]; then
                export PORT=80
fi

export PATH="/opt/python/3.7.15/bin:${PATH}"
echo 'export VIRTUALENVIRONMENT_PATH="/tmp/8dadf7c488f5f76/antenv"' >> ~/.bashrc
echo '. antenv/bin/activate' >> ~/.bashrc
PYTHON_VERSION=$(python -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))")
echo Using packages from virtual environment 'antenv' located at '/tmp/8dadf7c488f5f76/antenv'.
export PYTHONPATH=$PYTHONPATH:"/tmp/8dadf7c488f5f76/antenv/lib/python$PYTHON_VERSION/site-packages"
echo "Updated PYTHONPATH to '$PYTHONPATH'"
. antenv/bin/activate
GUNICORN_CMD_ARGS="--timeout 600 --access-logfile '-' --error-logfile '-' -c /opt/startup/gunicorn.conf.py --chdir=/tmp/8dadf7c488f5f76 -k uvicorn.workers.UvicornWorker" gunicorn app:app
