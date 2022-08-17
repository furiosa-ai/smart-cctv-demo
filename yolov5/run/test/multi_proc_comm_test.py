import argparse

from multiprocessing import Process
from multiprocessing.connection import Listener, Client


def server_main(port):
    address = ('localhost', port)
    listener = Listener(address)
    conn = listener.accept()

    while True:
        msg = conn.recv()
        if msg == 'close':
            conn.close()
            print("Recv msg")
            break
    listener.close()

    print("Server done")


def client_main(port):
    address = ('localhost', port)
    conn = Client(address)
    conn.send('close')
    print("Sent msg")
    conn.close()

    print("Client done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()

    server_proc = Process(target=server_main, args=(args.port,))
    client_proc = Process(target=client_main, args=(args.port,))

    server_proc.start()
    client_proc.start()

    server_proc.join()
    client_proc.join()

    print("Done")


if __name__ == "__main__":
    main()
