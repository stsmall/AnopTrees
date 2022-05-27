# conftest.py 
# put fixtures here and pytest runs can import them

import pytest
import requests

@pytest.fixture()
def disable_network_calls():
    ...
    