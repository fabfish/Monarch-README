import re
import matplotlib.pyplot as plt

# Train: 0 [   0/10009 (  0%)]  Loss:  6.929471 (6.9295)  Time: 24.915s,    5.14/s  (24.915s,    5.14/s)  LR: 1.000e-06  Data: 19.697 (19.697)

with open("vitae-monarch.txt","r") as f:
    lines = f.readlines()
    epochs = 0
    epoch_monarch = []
    loss_monarch = []
    loss_monarch_t = []
    time_monarch = []
    a1_monarch = []
    a5_monarch = []
    test_time_monarch = []
    time_monarch_x = []
    sum_time_monarch_x = 0
    epoch_time_monarch_x = []
    for line in lines:
        if line.startswith('Train'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            train_id, epoch_i, epoch_n, epoch_p, loss, loss_avg, time, speed, time_avg, speed_avg, lr1, lr2, data, data_avg = s
            if train_id>= 10:
                break
            
            if epoch_i % 50 == 0 and epoch_i!=0:
                sum_time_monarch_x += time
                time_monarch_x.append(sum_time_monarch_x)
                loss_monarch_t.append(loss_avg)
            if train_id<= 7:
                if epoch_i == 10000:
                    # epochs += epoch_i
                    epoch_monarch.append(train_id)
                    loss_monarch.append(loss_avg)
                    time_monarch.append(time_avg)
                    epoch_time_monarch_x.append(sum_time_monarch_x)
        if line.startswith('Test:'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            # Test: [ 390/390]  Time: 0.576 (1.866)  Loss:  5.1202 (3.7421)  Acc@1:  6.2500 (26.7060)  Acc@5: 20.0000 (51.1700)
            test_b, test_390, test_time, test_time_avg, loss, loss_avg, a1_l, a1, a1_avg, a5_l, a5, a5_avg = s
            if train_id>= 8:
                break
            if test_b == 390:
                a1_monarch.append(a1_avg)
                a5_monarch.append(a5_avg)
                test_time_monarch.append(test_time_avg)

with open("vitae-mlp.txt","r") as f:
    lines = f.readlines()
    epochs = 0
    # epoch_monarch = []
    loss_y = []
    time_y = []
    a1_y = []
    a5_y = []
    test_time_y = []
    test_time_mlp = []
    time_mlp_x = []
    sum_time_mlp_x = 0
    loss_mlp_t = []
    epoch_time_mlp_x = []
    for line in lines:
        if line.startswith('Train'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            # print(s)
            train_id, epoch_i, epoch_n, epoch_p, loss, loss_avg, time, speed, time_avg, speed_avg, lr1, lr2, data, data_avg = s
            # if train_id>= 8 or (epoch_i % 50 != 0):
            if train_id>= 8:
                break
            if epoch_i == 10000:
                # epochs += epoch_i
                # epoch_monarch.append(train_id)
                loss_y.append(loss_avg)
                time_y.append(time_avg)
                epoch_time_mlp_x.append(sum_time_mlp_x)
            if epoch_i % 50 == 0 and epoch_i!=0:
                sum_time_mlp_x += time
                time_mlp_x.append(sum_time_mlp_x)
                loss_mlp_t.append(loss_avg)
        if line.startswith('Test:'):
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            # Test: [ 390/390]  Time: 0.576 (1.866)  Loss:  5.1202 (3.7421)  Acc@1:  6.2500 (26.7060)  Acc@5: 20.0000 (51.1700)
            test_b, test_390, test_time, test_time_avg, loss, loss_avg, a1_l, a1, a1_avg, a5_l, a5, a5_avg = s
            if train_id>= 8:
                break
            if test_b == 390:
                a1_y.append(a1_avg)
                a5_y.append(a5_avg)
                test_time_y.append(test_time_avg)

plt.subplot(2,3,1)
plt.xlabel("epoch")
plt.ylabel("train loss",rotation=0)
l1=plt.plot(epoch_monarch, loss_monarch,color='g', label = 'ViTAE-monarch')
l2=plt.plot(epoch_monarch, loss_y,color='r', label = 'ViTAE-mlp')
plt.legend(['ViTAE-monarch','ViTAE-mlp'])

plt.subplot(2,3,2)
plt.xlabel("train time")
plt.ylabel("train loss",rotation=0)
plt.plot(time_monarch_x, loss_monarch_t,color='g')
plt.plot(time_mlp_x, loss_mlp_t,color='r')
plt.legend(['ViTAE-monarch','ViTAE-mlp'])

plt.subplot(2,3,3)
plt.xlabel("epoch")
plt.ylabel("train time",rotation=0)
plt.plot(epoch_monarch, time_monarch,color='g')
plt.plot(epoch_monarch, time_y,color='r')
plt.legend(['ViTAE-monarch','ViTAE-mlp'])

plt.subplot(2,3,4)
plt.xlabel("epoch")
plt.ylabel("test Acc",rotation=0)
l1=plt.plot(epoch_monarch, a5_monarch,color='g', label = 'ViTAE-monarch')
l2=plt.plot(epoch_monarch, a5_y,color='r', label = 'ViTAE-mlp')
l1=plt.plot(epoch_monarch, a1_monarch,color='b', label = 'ViTAE-monarch')
l2=plt.plot(epoch_monarch, a1_y,color='y', label = 'ViTAE-mlp')
plt.legend(['ViTAE-monarch@5','ViTAE-mlp@5','ViTAE-monarch@1','ViTAE-mlp@1'])

plt.subplot(2,3,5)
plt.xlabel("train time")
plt.ylabel("test Acc",rotation=0)
l1=plt.plot(epoch_time_monarch_x, a5_monarch,color='g', label = 'ViTAE-monarch')
l2=plt.plot(epoch_time_mlp_x, a5_y,color='r', label = 'ViTAE-mlp')
l3=plt.plot(epoch_time_monarch_x, a1_monarch,color='b', label = 'ViTAE-monarch')
l4=plt.plot(epoch_time_mlp_x, a1_y,color='y', label = 'ViTAE-mlp')
plt.legend(['ViTAE-monarch@5','ViTAE-mlp@5','ViTAE-monarch@1','ViTAE-mlp@1'])

plt.subplot(2,3,6)
plt.xlabel("epoch")
plt.ylabel("test time",rotation=0)
plt.plot(epoch_monarch, test_time_monarch,color='g')
plt.plot(epoch_monarch, test_time_y,color='r')
plt.legend(['ViTAE-monarch','ViTAE-mlp'])

plt.show()