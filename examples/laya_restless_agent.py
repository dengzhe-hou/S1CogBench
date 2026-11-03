def create_agent(device):
    from src.restless_models import LayaRestless
    return LayaRestless(device)
