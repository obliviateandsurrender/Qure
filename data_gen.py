import csv
data_col = list()
# 0 for benign, 1 for malignant
label = list()
with open('../Dataset/breast-cancer-wisconsin.data', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row[-1] == '4':
            label.append(1)
        elif row[-1] == '2':
            label.append(0)

        data = list()
        row = row[1:-1] # cut the label and the timestamp
        for ele in row:
            if ele == "?":
                print(row)
                data.append(-1)
            else:
                data.append(int(ele))
        data_col.append(data)


from sklearn import svm

X = data_col
y = label
model = svm.SVC(kernel='linear', C=1)
#model = svm.SVC(kernel='rbf')
model.fit(X, y)



support_vectors = model.coef_[0] #model.support_vectors_
dual_coefficients = model.intercept_ #model.dual_coef_

misclassified = list()
predict = model.predict(data_col)
print("dimension: ", len(data_col[0]))
print(data_col[0])
for i in range(len(label)):
    #print(model.predict(data_col))
    #print(predict[i])
    #print(label[i])
    if predict[i] != label[i]:
        misclassified.append(i)
print(misclassified)

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use PCA to project data onto 3 dimensions
pca = PCA(n_components=3)
assert len(data_col[0]) == 9
data_3d = pca.fit_transform(data_col)
assert len(data_3d[0]) == 3
# Plot the data in 3D space

w_3d = pca.transform(model.coef_)
print(model.intercept_)
#inter_3d = pca.transform(model.intercept_)
#w_3d = pca.transform(model.support_vectors_)

benign_3d = list()
mal_3d = list()
for i in range(len(label)):
    if label[i] == 0:
        benign_3d.append(data_3d[i])
    elif label[i] == 1:
        mal_3d.append(data_3d[i])
benign_3d = np.array(benign_3d)
mal_3d = np.array(mal_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#xx, yy = np.meshgrid(np.linspace(-5, 0.5, 15), np.linspace(-7.5, 0.5, 10))
#z = (-w_3d[0, 0] * xx - w_3d[0, 1] * yy) / w_3d[0, 2]
#ax.plot_surface(xx, yy, z, alpha=0.2, color='black')

# Compute decision function on a grid of points
xx, yy = np.meshgrid(np.linspace(-5, 0.5, 15), np.linspace(-10, 0.5, 15))
#Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = (-w_3d[0, 0] * xx - w_3d[0, 1] * yy - model.intercept_) / w_3d[0, 2]
'''
ax.plot_surface(xx, yy, z, alpha=0.2, color='black')

ax.scatter(benign_3d[:, 0], benign_3d[:, 1], benign_3d[:, 2], c='b', marker='o')
ax.scatter(mal_3d[:, 0], mal_3d[:, 1], mal_3d[:, 2], c='r', marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
'''

####calculate the distance
#### separate into 3 stages
stage_1 = list()
stage_2 = list()
stage_3 = list()
stage_1_lb = list()
stage_2_lb = list()
stage_3_lb = list()

max_dis = -1
min_dis = np.inf
count = 0
for i in range(len(data_col)):
        
    dist = model.decision_function([data_col[i]])[0]

    if label[i] ==1:
        if dist > max_dis:
            max_dis = dist
        if dist < min_dis:
            min_dis = dist

    if dist < 0:
        stage_1.append(data_col[i])
        stage_1_lb.append(label[i])
    else:
        if dist < 3: # further most benign sample
            stage_2.append(data_col[i])
            stage_2_lb.append(label[i])
        else:
            stage_3.append(data_col[i])
            stage_3_lb.append(label[i])

print("number of data in stage 1, ", len(stage_1))
print("number of data in stage 2, ", len(stage_2))
print("number of data in stage 3, ", len(stage_3))

'''
print(max_dis)
print(min_dis)

count = 0
for i in range(len(stage_2_lb)):
    if stage_2_lb[i] == 0:
        count += 1
print(count)
'''

#### randomly combine 3 stages
import random
time_series_dataset = list()

for i in range(6000):
    first_stage_end = int(random.uniform(0, 1) * 20)
    sec_stage_end = int(random.uniform(first_stage_end, 20))
    time_series = list()

    for time_step in range(first_stage_end):
        idx1 = int(random.uniform(0, 1) * len(stage_1))
        time_series.append(stage_1[idx1] + [stage_1_lb[idx1]])
    
    for time_step in range(first_stage_end, sec_stage_end):
        idx2 = int(random.uniform(0, 1) * len(stage_2))
        time_series.append(stage_2[idx2] + [stage_2_lb[idx2]])
    
    for time_step in range(sec_stage_end, 20):
        idx3 = int(random.uniform(0, 1) * len(stage_3))
        time_series.append(stage_3[idx3] + [stage_3_lb[idx3]])
    
    time_series_dataset.append(time_series)

time_series_dataset = np.array(time_series_dataset)
#np.savetxt('dataset.csv', time_series_dataset.reshape(6000, -1), delimiter=',')
np.save('dataset.npy', time_series_dataset)


'''
# Generate some sample data
data_check = np.transpose(np.array(data_col))
#print(data_check.shape)

# Create a box plot
plt.boxplot(list(data_check))

# Set the title and axis labels
plt.title('Box Plot')
plt.xlabel('Sample')
plt.ylabel('Value')

# Show the plot
plt.show()

count_9th = [ 0 for i in range(11)]
for i in range(len(data_col)):
    #print(data_col[i][8])
    count_9th[data_col[i][8]] += 1
print(count_9th[1:])
'''



