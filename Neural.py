import tensorflow as tf
import numpy as np 
import cv2

(x_train , y_train) ,(x_test , y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train , x_test = x_train/255.0 , x_test/255.0
def relu_prime(x):
	return np.where(x>0,1,0)
def relu(x):
	x[x<0] = 0
	return x

def empty():
	pass
def softmax(x):
	exps = np.exp(x-x.max())
	return exps/exps.sum()


class NeuralNetwork():
	

	#initializing our neural network
	def __init__(self, arch):
		self.L = arch.size -1

		self.parameter = {}
		self.derivative = {}

		self.parameter['a0'] = np.ones((arch[0],1))

		for l in range(1,self.L+1):
			self.parameter['a'+str(l)] = np.ones((arch[l] ,1))
			self.parameter['w'+str(l)] = np.random.randn(arch[l] , arch[l-1])*0.01
			self.parameter['z'+str(l)] = np.ones((arch[l],1))
			self.parameter['b'+str(l)] = np.ones((arch[l],1))


	def FeedForwardPropagation(self,x):

		#flatten

		x = np.reshape(x , (784,1))
		self.parameter['a0'] = x

		#dense and relu
		self.parameter['z1'] = np.add(np.dot(self.parameter['w1'],self.parameter['a0']),self.parameter['b1'])
		self.parameter['a1'] = relu(self.parameter['z1'])


		#dense and softmax

		self.parameter['z2'] = np.add(np.dot(self.parameter['w2'],self.parameter['a1']),self.parameter['b2'])
		self.parameter['a2'] = softmax(self.parameter['z2'])



	# def computeCrossentropy()
	# 	self.parameater['cost'] =y*np.log(self.parameter['a2'])
	

	def Backpropagation(self,y,alpha = 1e-2):

		self.derivative['dz2'] = self.parameter['a2']-y
		self.derivative['dw2'] = np.dot(self.derivative['dz2'],np.transpose(self.parameter['a1']))
		self.derivative['db2'] = self.derivative['dz2']

		self.derivative['dz1']= np.dot(np.transpose(self.parameter['w2']),self.derivative['dz2'])*relu_prime(self.parameter['a1'])
		self.derivative['dw1'] = np.dot(self.derivative['dz1'],np.transpose(self.parameter['a0']))
		self.derivative['db1'] = self.derivative['dz1']

		#upgrade value

		self.parameter['w1'] = self.parameter['w1'] - alpha*self.derivative['dw1']
		self.parameter['b1'] = self.parameter['b1'] - alpha*self.derivative['db1']

		self.parameter['w2'] = self.parameter['w2'] - alpha*self.derivative['dw2']
		self.parameter['b2'] = self.parameter['b2'] - alpha*self.derivative['db2']


	def predict(self,x):
		self.FeedForwardPropagation(x)
		return self.parameter['a2']


	def fit(self,X,Y,num_iter):
	

		for iter in range(num_iter):

			acc = 0

			for i in range(X.shape[0]):

				y_array = np.zeros((10,1))
				y = Y[i]
				x = X[i]
				for j in range(10):

					if j == y:
						y_array[j] = 1

				self.FeedForwardPropagation(x)
				self.Backpropagation(y_array)
				y_pred = self.predict(x)

				y_pred = y_pred.tolist()
				z = max(y_pred)
				yInd = y_pred.index(z)

				if yInd == y:
					acc+=1


			print('accuracy :' ,(acc/X.shape[0])*100 , 'no.:', iter)
			


	def processImage(self):

		cap = cv2.VideoCapture(0)
		cv2.namedWindow("TrackBars")
		cv2.resizeWindow("TrackBars" ,640,280)
		cv2.createTrackbar("Hue Min" , "TrackBars" , 0,179,empty)
		cv2.createTrackbar("Hue Max" , "TrackBars" , 179,179,empty)
		cv2.createTrackbar("Sat Min" , "TrackBars" , 0,255,empty)
		cv2.createTrackbar("Sat Max" , "TrackBars" , 255,255,empty)
		cv2.createTrackbar("Val Min" , "TrackBars" , 0,255,empty)
		cv2.createTrackbar("Val Max" , "TrackBars" , 255,255,empty)

		while True:
			success , img = cap.read()

			h_min = cv2.getTrackbarPos("Hue Min" , "TrackBars")
			h_max = cv2.getTrackbarPos("Hue Max" , "TrackBars")
			s_min = cv2.getTrackbarPos("Sat Min" , "TrackBars")
			s_max = cv2.getTrackbarPos("Sat Max" , "TrackBars")
			v_min = cv2.getTrackbarPos("Val Min" , "TrackBars")
			v_max = cv2.getTrackbarPos("Val Max" , "TrackBars")
			lower = np.array([h_min ,s_min ,v_min])
			upper = np.array([h_max , s_max , v_max])
			mask = cv2.inRange(cv2.cvtColor(img ,cv2.COLOR_BGR2HSV) ,lower,upper)
			image_result = cv2.bitwise_and(img , img , mask = mask)
			image = cv2.cvtColor(image_result ,cv2.COLOR_BGR2GRAY)
			image = cv2.resize(image ,(28,28) , interpolation = cv2.INTER_AREA)
			y_pred = self.predict(image)
			y_pred = y_pred.tolist()
			z = max(y_pred)
			yInd = y_pred.index(z)

			print(yInd)

			if(yInd == 0):
				cv2.putText(img , "T-Shirt" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 1):
				cv2.putText(img , "Trouser" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 2):
				cv2.putText(img , "Pullover" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 3):
				cv2.putText(img , "Dress" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 4):
				cv2.putText(img , "Coat" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 5):
				cv2.putText(img , "Sandal" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 6):
				cv2.putText(img, "Shirt" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 7):
				cv2.putText(img , "Sneaker" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 8):
				cv2.putText(img , "Bag" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)
			if(yInd == 9):
				cv2.putText(img , "Ankel" , (15,15) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,150,0),1)									

			cv2.imshow("img",img)
			cv2.imshow("image",image)
			cv2.waitKey(16)



ann = NeuralNetwork(np.array([784,128,10]))
ann.fit(x_train , y_train , 5)
ann.processImage()
