from binance.spot import Spot as Client

def create_client(api_key, api_secret):
    return Client(api_key, api_secret)