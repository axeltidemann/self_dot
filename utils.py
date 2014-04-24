def net_rmse(brain, signals):
    import Oger
    rmse = []

    for net, scaler in brain:
        scaled_signals = scaler.transform(signals)
        rmse.append(Oger.utils.rmse(net(scaled_signals[:-1]), scaled_signals[1:]))

    return rmse

class Parser:
    def __init__(self, learn_state, respond_state):
        self.learn_state = learn_state
        self.respond_state = respond_state
        
    def parse(self, message):
        print '[self.] received:', message
        if message == 'learn':
            self.learn_state.put(True)
        if message == 'respond':
            self.respond_state.put(True)
