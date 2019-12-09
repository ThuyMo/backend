import cv2
from django.http import JsonResponse
from rest_framework.decorators import api_view
from server import utils
from keras import backend as K
import base64
@api_view(['POST'])
def save_image(request):
    try:
        cmnd = (request.data['cmnd'])
        img=base64.b64decode(cmnd)
        filename='cmnd.jpg'
        with open(filename,'wb') as f:
            f.write(img)
        utils.rotate('cmnd.jpg')
        return JsonResponse({
            'err': 0,
            'msg': 'Xu ly anh thanh cong',
            'dt': 'lưu thành công',
        }, safe=False)
    except Exception as exc:
        return JsonResponse({
            'err': 1,
            'msg': f'Xu ly anh bi loi: {exc}',
            'dt': None,
        }, safe=False)



@api_view(['POST'])
def process_image(request):
    try:
        np_image=cv2.imread('cmnd.jpg')
        mat = (request.data['image'])
        img = base64.b64decode(mat)
        filename = 'mat.jpg'
        with open(filename, 'wb') as f:
            f.write(img)
        np_image1=cv2.imread('mat.jpg')
        filenames = [np_image,np_image1]
        # get embeddings file filenames
        embeddings = utils.get_embeddings(filenames)
        # define sharon stone
        sharon_id = embeddings[0]
        # verify known photos of sharon
        a=utils.is_match(embeddings[0], embeddings[1])
        kq=[]
        if (a==0):
            kq='Khuôn mặt không khớp'
        else:
            utils.cmnd_test('cmnd_test.jpg')
            utils.detect_cmnd('cmnd_test.jpg','cmnd_detect.jpg')
            # utils.rotated('cmnd_cut.jpg','cmnd_rotate.jpg')
            kq=utils.detect_text('cmnd_detect.jpg')

        K.clear_session()
        return JsonResponse({
            'err': 0,
            'msg': 'Xu ly anh thanh cong',
            'dt': kq,
        }, safe=False)

    except Exception as exc:
        return JsonResponse({
            'err': 1,
            'msg': f'Xu ly anh bi loi: {exc}',
            'dt': None,
        }, safe=False)


# @api_view(['POST'])
# def detect_text(request):
#     try:
#
#         np_image=cv2.imread('cmnd.jpg')
#         utils.detect_cmnd(np_image,'cmnd_cut.jpg')
#         utils.rotated('cmnd_cut.jpg','cmnd_rotate.jpg')
#         result=utils.detect_text('cmnd_rotate.jpg')
#
#         K.clear_session()
#         return JsonResponse({
#             'err': 0,
#             'msg': 'Xu ly dư lieu thanh cong',
#             'dt':result,
#
#         }, safe=False)
#
#     except Exception as exc:
#         return JsonResponse({
#             'err': 1,
#             'msg': f'Xu ly du lieu bi loi: {exc}',
#             'dt': None,
#         }, safe=False)
#
#


@api_view(['POST'])
def login(request):
    return JsonResponse({
        'err': 0,
        'msg': 'Dang nhap thanh cong',
        'dt': {
            'username': 'Abc',
            'password': '12345',
        }
    }, safe=False)
