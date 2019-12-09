import numpy as np
import cv2
from  matplotlib  import  pyplot
from  PIL  import  Image
from  numpy  import  asarray
from  scipy.spatial.distance  import  cosine
from mtcnn.mtcnn import MTCNN
# from keras_applications.imagenet_utils import _obtain_input_shape
from  keras_vggface.vggface  import  VGGFace
from  keras_vggface.utils  import  preprocess_input
import ssl
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image, ImageFilter,ImageEnhance
import imutils
from tesserocr import PyTessBaseAPI
import tempfile
from string import digits
import re
import pandas as pd
from datetime import date

def rotate(filename):
    image = cv2.imread(filename)
    rotated=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow( rotated)
    cv2.imwrite('cmnd_test.jpg',rotated)
def equalizeHist(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    gray = cv2.equalizeHist(gray)
    cv2.imwrite('cmnd_test.jpg',gray)
def detect_chu(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle


    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imwrite('cmnd_test.jpg',rotated)

def detect_cmnd(filename,output):
    image = cv2.imread(filename)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    startY=screenCnt[0][0][0]
    startX=screenCnt[0][0][1]
    endY=screenCnt[1][0][1]
    endX=screenCnt[3][0][0]
    im=image[startY:endY, startX:endX]
    plt.imshow(im)
    cv2.imwrite(output,im)
    # cv2.imshow('image_all',im)
    # cv2.waitKey(0)


def kiemtratinh(tinhthanh,tinh):
    for i in range(len(tinhthanh)):
        a=tinhthanh[i].find(tinh)
        if a==0:
            return 1
#keras==2.2.5
# tensorflow== 1.14
def multipartimage_to_numpyimage(multipartfile):
    image_bytedata = multipartfile.read()

    decoded = np.frombuffer(image_bytedata, np.uint8)
    np_image = cv2.imdecode(decoded, -1)

    return np_image

def extract_face(pixels, required_size=(224, 224)):

    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1: y2, x1: x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    # cv2.imwrite('a.jpg',face_array)
    return face_array


def get_embeddings(filenames):
    ssl._create_default_https_context = ssl._create_unverified_context

    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if (score <= thresh):
        return 1
    else:
        return 0

def rotated(filename,file_out):
    image = cv2.imread(filename)
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))
    # plt.imshow(image)
    cv2.imwrite(file_out, rotated)


def detect_cmnd(image,filename_out):
    # image = cv2.imread('cmnd_test.jpg')
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # ratio = image.shape[0] / 300.0
    # orig = image.copy()
    # image = imutils.resize(image, height = 300)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    startY = screenCnt[0][0][1]
    startX = screenCnt[1][0][0]
    endY = screenCnt[2][0][1]
    endX = screenCnt[3][0][0]
    im = image[startY:endY, startX:endX]
    # plt.imshow(im)
    # cv2.waitKey(0)
    cv2.imwrite(filename_out, im)


IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


size = None


def get_size_of_scaled_image(im):
    global size
    if size is None:
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
    return size
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new

def detect_text(filename):

    image = cv2.imread(filename)
    img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 5, 13)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.imwrite('cmnd1.jpg', img)
    image = cv2.imread('cmnd1.jpg')
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # que quan
    x = 300
    y = 368
    w = 600
    h = 100

    nativecountry = image[y:y + h, x:x + w]

    ret1, th1 = cv2.threshold(nativecountry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)

    cv2.imwrite('quequan.jpg', th1)

    # name
    x = 220
    y = 190
    w = 630
    h = 100
    name = image[y:y + h, x:x + w]

    ret2, th2 = cv2.threshold(name, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)

    # th2 = cv2.adaptiveThreshold(th2.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,3)
    # th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    # th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('name.jpg', th2)

    # birthday

    x = 270
    y = 310
    w = 630
    h = 60
    birthday = image[y:y + h, x:x + w]
    ret3, th3 = cv2.threshold(birthday, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)

    cv2.imwrite('birthday.jpg', th3)

    # dia chi

    x = 280
    y = 470
    w = 640
    h = 110

    address = image[y:y + h, x:x + w]

    # address = cv2.bitwise_not(address)
    thresh = cv2.threshold(address, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)[1]
    # thresh = cv2.bitwise_not(thresh)
    ret4, th4 = cv2.threshold(address, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)

    cv2.imwrite('address.jpg', thresh)

    # so cmnd
    x = 460
    y = 145
    w = 330
    h = 50

    number = image[y:y + h, x:x + w]

    ret5, th5 = cv2.threshold(number, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_TRUNC)

    cv2.imwrite('sochungminh.jpg', th5)

    result=[]
    images = ['address.jpg', 'name.jpg', 'birthday.jpg', 'quequan.jpg', 'sochungminh.jpg']
    with PyTessBaseAPI(path='D:/tessdata-master/.', lang='vie') as api:
        for img in images:
            api.SetImageFile(img)
            a=api.GetUTF8Text()
            result.append(a)
    kq = []
    print(kq)
    dc=result[0].replace('\n', ' ')
    ten = result[1].replace('\n', ' ')
    ngay = result[2].replace('\n', ' ')
    que = result[3].replace('\n', ' ')
    socmnd = result[4].replace('\n', ' ')
    df = pd.read_excel(r'tinh.xlsx',
                       encoding='utf-8')  # for an earlier version of Excel, you may need to use the file extension of 'xls'
    tinhthanh = df['Tỉnh']
    st = result[0]
    t = st.split('\n')
    dc = t[1]
    dc = dc.split(', ')
    tinh = dc[2]
    if kiemtratinh(tinhthanh, tinh) == 1:
        NGAYSINH = result[2]
        NGAYSINH = NGAYSINH.split('-')
        ngay = NGAYSINH[0].split(' ')[2]
        ngay = int(ngay)
        thang = NGAYSINH[1]
        thang = int(thang)
        nam = NGAYSINH[2].split(' ')
        nam = nam[0]
        nam = int(nam)

        today = date.today()
        year = today.strftime('%m/%d/%Y')
        a = year[6:10]
        a = int(a)
        if (((a - nam) >= 15) & ((a - nam) <= 150)) & ((ngay >= 1) & (ngay <= 31)) & ((thang >= 1) & (thang <= 12)):
            kq = 'Đăng ký thành công'
        else:
            kq = 'Thông tin sai sót, đề nghị nhập lại'
    else:
        kq = 'Không tồn tại tỉnh'

    return kq