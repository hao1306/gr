rnn.py is the newest version of network file

There are four types of Netwroks:

- CNN + LSTM, process x, y, yaw, Input: batchsize * 4 * 3
- CNN + LSTM, process Images, x, y, yaw, Input: batchsize * 4 * Images (3 * 800 * 750) and batchsize * 4 * 3
- CNN, process x, y, yaw, Input: batchsize * 4 * 3
- CNN, process Images, x, y, yaw, Input: batchsize * 4 * Images (3 * 800 * 750) and batchsize * 4 * 3
