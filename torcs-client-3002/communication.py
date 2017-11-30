from multiprocessing.connection import Listener
from multiprocessing.connection import Client


class Receiver():

    def __init__(self, port):
        self.address = ('localhost', port)

    def listen(self):
        listener = Listener(self.address, authkey=b'secret password')
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        msg = conn.recv()
        conn.close()
        listener.close()
        return msg

        #     print(msg)
        #     if msg == 'close':
        #         conn.close()

        #         break
        # listener.close()
        # return msg
    def listenSend(self, message):
        listener = Listener(self.address, authkey=b'secret password')
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        msg = conn.recv()
        conn.send(message)
        conn.close()
        listener.close()
        return msg


class Sender():

    def __init__(self, port):
        self.address = ('localhost', port)

    def send(self, message):
        conn = Client(self.address, authkey=b'secret password')
        conn.send(message)
        conn.close()

    def sendReceive(self, message):
        conn = Client(self.address, authkey=b'secret password')
        conn.send(message)
        msg = conn.recv()
        print(msg)
        conn.close()
        return msg
        