import sys

from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtGui

from PyQt5.QtGui import QImage,QPixmap

import PIL.Image
from PIL import Image
from PIL import ImageOps as imgOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import random
from sklearn.metrics import accuracy_score,classification_report





class ArtificialNeuralNetwork():

    model = keras.models.load_model('saved_model/')

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    def Augment_Images(x_train=X_train, x_test=X_test):
        for i in range(60000):

            if (random.randint(0, 1) == 0):
                x_train[i] = imgOps.autocontrast(Image.fromarray(x_train[i]))
                for x in range(28):
                    for y in range(28):
                        if 0 <= x_train[i][x][y] < 25:
                            x_train[i][x][y] = int(random.uniform(50, 100))
                        elif 225 < x_train[i][x][y] <= 255:
                            x_train[i][x][y] = int(random.uniform(155, 200))
            if (random.randint(0, 1) == 0):
                x_train[i] = imgOps.invert(Image.fromarray(x_train[i]))
            if (random.uniform(0, 1) > 0.30):
                shear = random.uniform(-7.0, 7.0)
                tXY = random.uniform(-12.5, 12.5)
                zXY = random.uniform(1.3, 2.5)
                x_train[i] = ImageDataGenerator().apply_transform(x=x_train[i].reshape(28, 28, 1),
                                                                  transform_parameters={'tx': tXY, 'ty': tXY, 'zx': zXY,
                                                                                        'zy': zXY,
                                                                                        'shear': shear}).reshape(28, 28)
            # Image.fromarray(X_train[i]).save("Eğitim Veri Seti/savedImage{}.jpg".format(str(i)))
            x_train[i] = np.asarray(x_train[i])

        for i in range(10000):

            if (random.randint(0, 1) == 0):
                x_test[i] = imgOps.autocontrast(Image.fromarray(x_test[i]))
                for x in range(28):
                    for y in range(28):
                        if 0 <= x_test[i][x][y] < 25:
                            x_test[i][x][y] = int(random.uniform(50, 100))
                        elif 225 < x_test[i][x][y] <= 255:
                            x_test[i][x][y] = int(random.uniform(155, 200))
            if (random.randint(0, 1) == 0):
                x_test[i] = imgOps.invert(Image.fromarray(x_test[i]))
            if (random.uniform(0, 1) > 0.30):
                shear = random.uniform(-7.0, 7.0)
                tXY = random.uniform(-12.5, 12.5)
                zXY = random.uniform(1.3, 2.5)
                x_test[i] = ImageDataGenerator().apply_transform(x=x_test[i].reshape(28, 28, 1),
                                                                 transform_parameters={'tx': tXY, 'ty': tXY, 'zx': zXY,
                                                                                       'zy': zXY,
                                                                                       'shear': shear}).reshape(28, 28)
            # Image.fromarray(X_test[i]).save("Test Veri Seti/savedImage{}.jpg".format(str(i)))
            x_test[i] = np.asarray(x_test[i])


    def model_evaluation(model=model,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test):

        y_predicted = model.predict(X_test)

        y_predicted_labels = [np.argmax(i) for i in y_predicted]

        print("Test Accuracy:", accuracy_score(y_test, y_predicted_labels))

        print("Test Report:", classification_report(y_test, y_predicted_labels))

        y_predicted = model.predict(X_train)

        y_predicted_labels = [np.argmax(i) for i in y_predicted]

        print("Train Accuracy:", accuracy_score(y_train, y_predicted_labels))

        print("Train Report:", classification_report(y_train, y_predicted_labels))


    def predict(img,model=model):
        img = imgOps.grayscale(img)
        img = img.resize((28, 28), resample=PIL.Image.LANCZOS)
        img = imgOps.autocontrast(img)
        img = np.asarray(img)
        img=img.reshape(28,28,1)
        img = img / 255

        predict = model.predict(np.array([img, ]))

        key = str(np.argmax(predict))
        predict=int(100*predict[0][int(key)])

        x = "Output " + key

        return x




class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        super(MainWindow, self).__init__()
        loadUi("untitled5.ui", self)
        self.w = None
        qpixmap=QPixmap("uploadIcon.png")
        self.imglabel.setPixmap(qpixmap)
        qpixmap = QPixmap("drawingIcon.png")

        self.imglabel_2.setPixmap(qpixmap)

        self.pushButton.clicked.connect(self.make_predict_photo)

        self.pushButton_2.clicked.connect(self.draw_digit)


    def make_predict_photo(self):
        self.label.clear()
        try:
            fileName=QFileDialog.getOpenFileName(self,"Upload Image",filter="PNG Files (*.png);;Jpg Files(*.jpg);;Jpeg Files(*.jpeg)")
        except:
            self.label.setText("IO Error")

        if str(fileName)!="('', '')":
            image = QImage(fileName[0])
            image = Image.fromqpixmap(image)
            print(type(image))
            self.label.setText(ArtificialNeuralNetwork.predict(image))


    def draw_digit(self):
        self.label.clear()
        if self.w is None:
            self.w=SubWindow(self)
        self.hide()
        self.w.show()


class SubWindow(QMainWindow):

    def __init__(self,m):
        QMainWindow.__init__(self)
        super(SubWindow, self).__init__()
        self.m=m
        loadUi("untitled4.ui", self)
        self.setFixedWidth(500)
        self.setFixedHeight(500)

        self.pen_color = QtGui.QColor('white')
        self.canvas_fill_color=QtGui.QColor('black')
        self.penSize=5
        canvas = QtGui.QPixmap(501,371)
        canvas.fill(QtGui.QColor('black'))
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.widget)

        self.last_x, self.last_y = None, None
        self.pushButton.clicked.connect(self.make_predict_drawing)
        self.pushButton_2.clicked.connect(self.clear)
        self.pushButton_3.clicked.connect(self.invert)
        self.dial.valueChanged.connect(lambda :self.dialer())
        self.dial.setMinimum(6)
        self.dial.setMaximum(15)



    def dialer(self):
        self.penSize=self.dial.value()


    def invert(self):

        if self.pen_color.value()==255:
            self.pen_color = QtGui.QColor('black')
            self.canvas_fill_color = QtGui.QColor('white')
            canvas = QtGui.QPixmap(501, 371)
            canvas.fill(self.canvas_fill_color)
            self.label.setPixmap(canvas)
            self.label2.clear()
        else:
            self.pen_color = QtGui.QColor('white')
            self.canvas_fill_color = QtGui.QColor('black')
            canvas = QtGui.QPixmap(501, 371)
            canvas.fill(self.canvas_fill_color)
            self.label.setPixmap(canvas)
            self.label2.clear()

    def clear(self):

        if self.pen_color.value()==255:
            self.pen_color = QtGui.QColor('white')
            self.canvas_fill_color = QtGui.QColor('black')
            canvas = QtGui.QPixmap(501, 371)
            canvas.fill(self.canvas_fill_color)
            self.label.setPixmap(canvas)
            self.label2.clear()
        else:
            self.pen_color = QtGui.QColor('black')
            self.canvas_fill_color = QtGui.QColor('white')
            canvas = QtGui.QPixmap(501, 371)
            canvas.fill(self.canvas_fill_color)
            self.label.setPixmap(canvas)
            self.label2.clear()


    def make_predict_drawing(self):

        image=self.label.pixmap().toImage()
        image=Image.fromqpixmap(image)
        self.label2.setText(ArtificialNeuralNetwork.predict(image))



    def closeEvent(self,event):
        self.canvas_fill_color = QtGui.QColor('black')
        canvas = QtGui.QPixmap(501, 371)
        canvas.fill(QtGui.QColor(self.canvas_fill_color))
        self.pen_color = QtGui.QColor('white')
        self.label.setPixmap(canvas)

        self.label2.clear()
        self.m.show()

    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())
        p = painter.pen()
        p.setWidth(self.penSize)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()


        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()


    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setFixedWidth(1223)
    window.setFixedHeight(643)
    window.show()

    try:
        sys.exit(app.exec())
    except:
        print("Exiting...")




if __name__ == '__main__':

    main()












