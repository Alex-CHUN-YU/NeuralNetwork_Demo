# -*- coding: utf-8 -*-
__author__ = "ALEX-CHUN-YU (P76064538@mail.ncku.edu.tw)"
import numpy as numpy

# 神經網路透過 back propagation
class Neural(object):

	def __init__(self):
		# input data 有四筆資料
		self.X = numpy.array([[0,1,1],
			[0,1,1],
			[1,1,1],
			[1,0,1]])
		# output data 預期輸出(四筆)
		self.y = numpy.array([[0],
			  [0],
			  [1],
			  [1]])
		# 隨機的權重
		numpy.random.seed(1)
		self.synapse0 = 2 * numpy.random.random((3, 4)) - 1
		self.synapse1 = 2 * numpy.random.random((4, 1)) - 1	

	# 非線性轉換(Sigmoid)和微分
	def nonline(self, x, derive = False):
		if (derive == True):
			return x * (1 - x)
		return 1 / (1 + numpy.exp(-x))

	# Back propagation
	def train(self):
		for i in range(10000):
			l1 = self.nonline(numpy.dot(self.X, self.synapse0))
			z2 = numpy.dot(self.X, self.synapse0)
			l1 = self.nonline(z2)
			z3 = numpy.dot(l1, self.synapse1)
			# 實際輸出
			l2 = self.nonline(z3)
			# 預期結果減掉實際結果
			l2_error = self.y - l2
			if (i % 1000) == 0:
				# 可以觀察到 error 也就是誤差值會越來越低
				print("Error:" + str(numpy.mean(numpy.abs(l2_error))))
			l2_delta = l2_error * self.nonline(l2, derive = True)
			l1_error = l2_delta.dot(self.synapse1.T)
			l1_delta = l1_error * self.nonline(l1, derive = True)
			self.synapse1 += l1.T.dot(l2_delta)
			self.synapse0 += self.X.T.dot(l1_delta)
		print("Output after traning")
		# 可看出實際輸出(經過訓練)與預期輸出因而接近
		print(l2)

if __name__ == "__main__":
	neural = Neural()
	neural.train()



