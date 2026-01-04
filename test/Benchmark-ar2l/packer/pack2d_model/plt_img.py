import matplotlib.pyplot as plt

# 定义点
points = [(0.0, 0.0), (3.089, 9.189), (162.592, 8.689), (165.889, -36.711), (2.089, -40.011), (0.0, 0.0)]
points_will = [(-2.089, 40.011), (1,49.2), (160.503, 48.7), (163.8, 3.3), (0 ,0), (-2.089,40.011)]


points_90 = [(0.0, 0.0), (-9.189, 3.089), (-8.689, 162.592), (36.711, 165.889), (40.011, 2.089), (0.0, 0.0)]

# 提取x和y坐标
x, y = zip(*points_will)

# 画图
plt.figure(figsize=(10, 10))
plt.scatter(x, y, color='red')
plt.plot(x, y, linestyle='-', color='blue', linewidth=2)
plt.title('Scatter Plot of Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
