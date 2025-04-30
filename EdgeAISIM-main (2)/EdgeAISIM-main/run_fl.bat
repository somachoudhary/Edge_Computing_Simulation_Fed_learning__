@echo off
start cmd /k python federated_learning\fl_server.py
timeout /t 5
start cmd /k python federated_learning\fl_client.py
