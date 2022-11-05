import re
import matplotlib.pyplot as plt

# Train: 0 [   0/10009 (  0%)]  Loss:  6.929471 (6.9295)  Time: 24.915s,    5.14/s  (24.915s,    5.14/s)  LR: 1.000e-06  Data: 19.697 (19.697)

with open("vitae-monarch.txt","r") as f:
    lines = f.readlines()
    epochs = 0
    epoch_x = []
    loss_x = []
    time_x = []
    for line in lines:
        if line.startswith('Train'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            # print(s)
            train_id, epoch_i, epoch_n, epoch_p, loss, loss_avg, time, speed, time_avg, speed_avg, lr1, lr2, data, data_avg = s
            # if train_id>= 8 or (epoch_i % 50 != 0):
            if train_id>= 8:
                break
            if epoch_i ==0 or epoch_i % 50 != 0:
                continue
            epochs += epoch_i
            epoch_x.append(epochs)
            loss_x.append(loss_avg)
            time_x.append(time_avg)

with open("vitae-mlp.txt","r") as f:
    lines = f.readlines()
    epochs = 0
    # epoch_x = []
    loss_y = []
    time_y = []
    for line in lines:
        if line.startswith('Train'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            # print(s)
            train_id, epoch_i, epoch_n, epoch_p, loss, loss_avg, time, speed, time_avg, speed_avg, lr1, lr2, data, data_avg = s
            # if train_id>= 8 or (epoch_i % 50 != 0):
            if train_id>= 8:
                break
            if epoch_i ==0 or epoch_i % 50 != 0:
                continue
            # epochs += epoch_i
            # epoch_x.append(epochs)
            loss_y.append(loss_avg)
            time_y.append(time_avg)

plt.subplot(1,2,1)
plt.xlabel("batch")
plt.ylabel("avg loss")
l1=plt.plot(epoch_x, loss_x,color='g', label = 'monarch')
l2=plt.plot(epoch_x, loss_y,color='r', label = 'mlp')
plt.legend(['monarch','mlp'])
plt.subplot(1,2,2)
plt.xlabel("batch")
plt.ylabel("avg time")
plt.plot(epoch_x, time_x,color='g')
plt.plot(epoch_x, time_y,color='r')
plt.legend(['monarch','mlp'])
plt.show()