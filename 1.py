with open('./result.txt','w')as f:
	f.write("status=finish\nOriginal_ModelAcc={}\nFinish_ModelAcc={}".format("0.97","0.67")) # 如果准确率(精度)可以获取到的话
	f.close()