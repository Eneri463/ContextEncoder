import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from random import randint
import cv2
import os
from imutils import paths

tf.disable_eager_execution()

batchSize = 6
imageHeight = 256
imageWidth = 256
imageChannels = 3
cropHeight = 64
cropWidth = 64 
epochs = 30
pp = 30000 # количество изображений, которые будут использоваться для обучения/тестирования
pathM = 'data/masks'
pathMT = 'data/masks_test'
pathI = 'data/CelebA'

# -------------------------------------------------------------------
# класс, создающий контекстный кодировщик и обучающий его
class ContextEncoder:
    
    def __init__(self,  trainData, testData, epochs, batchSize, imageHeight = 128, imageWidth = 128, imageChannels = 3, cropHeight = 64, cropWidth = 64):
        # ----- гиперпараметры обучения
        self.epochs = epochs        # количество эпох
        self.batchSize = batchSize      # размер одного батча
        self.imageHeight = imageHeight   # высота изображения
        self.imageWidth = imageWidth    # ширина изображения
        self.cropHeight = cropHeight     # высота восстанавливаемой области
        self.cropWidth = cropWidth      # ширина восстанавливаемой области
        self.imageChannels = imageChannels   # количество каналов в изображении
        
        # ----- объекты и данные, используемые при обучении
        
        # данные для оубчения
        self.trainData = trainData
        self.testData = testData
        # исходное изображение с повреждённой областью внутри
        self.inputs =  tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        # оригинальная восстанавливаемая часть
        self.cropInputs = tf.placeholder(tf.float32, [self.batchSize, self.cropHeight, self.cropWidth, self.imageChannels])
        # генератор
        generator = GEN("generator")
        # дискриминатор
        discriminator = DIS("discriminator")
        
        self.cropFake = generator(self.inputs)
        self.disTrue = discriminator(self.cropInputs)
        self.disFake = discriminator(self.cropFake)
        
        self.disLoss = -tf.reduce_mean(tf.log(self.disTrue + 1e-5) + tf.log(1 - self.disFake + 1e-5))
        self.genLoss = -tf.reduce_mean(tf.log(self.disFake + 1e-5)) + 100*tf.reduce_mean(tf.reduce_sum(tf.square(self.cropInputs - self.cropFake), [1, 2, 3]))
        
        self.disOptimizer = tf.train.AdamOptimizer(2e-4).minimize(self.disLoss, var_list=discriminator.get_var())
        self.genOptimizer = tf.train.AdamOptimizer(2e-4).minimize(self.genLoss, var_list=generator.get_var())
        
        self.costDis = tf.summary.scalar("disLoss", self.disLoss)
        self.costGen = tf.summary.scalar("genLoss", self.genLoss)
        self.merged = tf.summary.merge_all()
        self.writerTest = tf.summary.FileWriter("./logs/test")
        self.writerTrain = tf.summary.FileWriter("./logs/train")
        
        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
    # -------------------------------------------------------------- обучение
    def train(self, i = 0, whenSave = 1):
        
        tf.reset_default_graph()
        
        self.writerTrain.add_graph(self.sess.graph)
        self.writerTest.add_graph(self.sess.graph)
        
        for epoch in range(self.epochs):
            
            im = []
            im2 = []
            
            # ----- шаг обучения
            for numberBatch in range(len(self.trainData)):
                
                im, crop = self.trainData[numberBatch]
                
                self.sess.run(self.disOptimizer, feed_dict={self.inputs: im, self.cropInputs: crop})
                self.sess.run(self.genOptimizer, feed_dict={self.inputs: im, self.cropInputs: crop})
            
            # ----- вывод промежуточных результатов:
            im2, crop2 = self.testData[0]

            summaryTrain, resLossDTrain, resLossGTrain = self.sess.run([self.merged, self.disLoss, self.genLoss], feed_dict={self.inputs: im, self.cropInputs: crop})

            summaryTest, resLossDTest, resLossGTest = self.sess.run([self.merged, self.disLoss, self.genLoss], feed_dict={self.inputs: im2, self.cropInputs: crop2})

            self.writerTrain.add_summary(summaryTrain, i)
            self.writerTest.add_summary(summaryTest, i)

            print("Итерация " + str(i) + ". D loss = " + str(resLossDTrain) + ", D loss Test = " + str(resLossDTest) + ", G loss = " + str(resLossGTrain) + ", G loss Test = " + str(resLossGTest) + ".")

            # ----- сохраняем параметры нейросети каждые whenSave эпох
            if (epoch + 1) % whenSave == 0:

                resImage = self.sess.run([self.cropFake], feed_dict={self.inputs: im})
                Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results//" + str(i) + ".jpg")
                self.saver.save(self.sess, "./save_para//para.ckpt")

            self.trainData.on_epoch_end()
            self.testData.on_epoch_end()
            
    # -------------------------------------------------------------- 
    # восстановление данных        
    def restoreModel(self, pathMeta, path):

        self.saver = tf.train.import_meta_graph(pathMeta)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
    
    
    # -------------------------------------------------------------- 
    # использование готовой модели для восстановления изображения:
    def useModel(self, image):
        
        resImage = self.sess.run([self.outputs], feed_dict={self.damagedInputs: image, self.masks: mask})
        Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results.jpg")
        print("Результат сохранен")

# -------------------------------------------------------------------
# реализация слоёв нейронной сети 

# ----- свёртка
# подаётся: 
# название операции, входные данные, количество фильтров, размер ядра, шаг stride, тип паддинга
def conv2D(name, inputs, filters, kSize, stride, padding):
    
    with tf.variable_scope(name):
        
        W = tf.get_variable("W", shape=[kSize, kSize, inputs.shape[-1], filters], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        
        return  tf.math.add(tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding), b)

# ----- обратная свёртка
# подаётся: 
# название операции, входные данные, количество фильтров, размер ядра, шаг stride, тип паддинга
def unconv2D(name, inputs, filters, kSize, stride, padding, r = 2):
    
    with tf.variable_scope(name):
        
        w = tf.get_variable("W", shape=[kSize, kSize, filters, int(inputs.shape[-1])], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        
        B = tf.shape(inputs)[0]
        H = int(inputs.shape[1])
        W = int(inputs.shape[2])
        
        return  tf.math.add(tf.nn.conv2d_transpose(inputs, w, [B, H*r, W*r, filters], [1, stride, stride, 1], padding), b)

# ----- полносвязный слой / вектор 
def fullyConnected(name, inputs, filters):
    
    inputs = tf.layers.flatten(inputs)
    
    with tf.variable_scope(name):
        
        W = tf.get_variable("W", [int(inputs.shape[-1]), filters], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", [filters], initializer=tf.constant_initializer(0.))
    
        return tf.math.add(tf.matmul(inputs, W), b)


# -------------------------------------------------------------------
# класс генератора
class GEN:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            # ----- блок кодировщика (encoder)
            model = tf.nn.leaky_relu(conv2D("conv1", inputs, 64, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv2", model, 64, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv3", model, 128, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv4", model, 256, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv5", model, 512, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv6", model, 4000, 4, 2, "VALID"))
            
            
            # ----- блок декодировщика (decoder)
            model = tf.nn.relu(unconv2D("unconv1", model, 512, 4, 2, "VALID", 4))
            model = tf.nn.relu(unconv2D("unconv2", model, 256, 4, 2, "SAME"))
            model = tf.nn.relu(unconv2D("unconv3", model, 128, 4, 2, "SAME"))
            model = tf.nn.relu(unconv2D("unconv4", model, 64, 4, 2, "SAME"))
            model = tf.nn.tanh(unconv2D("unconv5", model, 3, 4, 2, "SAME"))
    
            return model

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

# -------------------------------------------------------------------
# класс дискриминатора
class DIS:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            # ----- блок кодировщика (encoder)
            model = tf.nn.leaky_relu(conv2D("conv1", inputs, 64, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv3", model, 128, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv4", model, 256, 4, 2, "SAME"))
            model = tf.nn.leaky_relu(conv2D("conv5", model, 512, 4, 2, "SAME"))
            model = tf.sigmoid(fullyConnected("fc", model, 1))
    
            return model

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


# -------------------------------------------------------------------
# класс, генерирующий тренировочные данные
class createAugment():
    
    # --
    # инициализация объекта класса
    def __init__(self, X, batch_size=32, dim=(128, 128), n_channels=3, crop_size = 64):
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.crop_size = crop_size
        self.indexes = []
        self.X = X
        self.on_epoch_end()
        
    # --
    # результат: максимальное количество батчей в одной эпохе
    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))
    
    # --
    # результат: взятие батча с заданным номером (индексом)
    def __getitem__(self, index):
        indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_inputs, y_output = self.data_generation(indexs)
        return X_inputs, y_output
    
    # --
    # функция, повторяющаяся в конце каждой эпохи
    # результат: новая совокупность индексов изображений для очередного батча
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        np.random.shuffle(self.indexes)
    
    # --
    # результат: батч данных, включающий в себя 
    # маскированное изображение, маску и исходное изображение
    def data_generation(self, idxs):
        
        # создаём пустые массивы для маскированных изображений батча, их масок 
        # и исходных изображений соответственно
        masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
        y_batch = np.empty((self.batch_size, 64, 64, 3)) # Original image
        
        for i, idx in enumerate(idxs):
            
            image_copy = self.X[idx].copy()
            masked_image, y_new  = self.createMask(image_copy)
            masked_images[i,] = masked_image/255
            y_batch[i] = y_new/255
            
        return masked_images, y_batch

    def createMask(self, image):
        
        x1 = randint(1, 64)
        y1 = randint(1, 64)
        x2 = x1 + 64
        y2 = y1 + 64
        
        image2 = image.copy()
        cropped_image = image2[y1:y2, x1:x2]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        return image, cropped_image


pp = 20000
paths = os.listdir('data/faces') # список файлов датасета
image = np.empty((pp, 128, 128, 3), dtype='uint8')

i = 0

for path in paths:
    img = Image.open(os.path.join('data/faces', path))
    img = img.resize((128,128))
    b = tf.keras.preprocessing.image.img_to_array(img)
    image[i] = b
    i = i+1
    if i == pp:
        break
        
trainData = createAugment(image[0:int(pp*0.9)], 20)
testData = createAugment(image[int(pp*0.9):], 20)

network = ContextEncoder(trainData, testData, epochs, batchSize, imageHeight, imageWidth, imageChannels, cropHeight, cropWidth)

network.train()