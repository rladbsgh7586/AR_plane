package com.google.ar.core.examples.java.host;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.view.SurfaceView;
import android.view.View;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

public class ImageHelper {
    public static byte[] imageToByteArray(Image image) {
        byte[] data = null;
        if (image.getFormat() == ImageFormat.JPEG) {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            data = new byte[buffer.capacity()];
            buffer.get(data);
            return data;
        } else if (image.getFormat() == ImageFormat.YUV_420_888) {
            data = NV21toJPEG(
                    rotateYUV420ToNV21(image),
                    image.getHeight(), image.getWidth());
        }
        return data;
    }

    private static byte[] rotateYUV420ToNV21(Image imgYUV420) {

        byte[] rez = new byte[imgYUV420.getWidth() * imgYUV420.getHeight() * 3 / 2];
        ByteBuffer buffer0 = imgYUV420.getPlanes()[0].getBuffer();
        ByteBuffer buffer1 = imgYUV420.getPlanes()[2].getBuffer();
        ByteBuffer buffer2 = imgYUV420.getPlanes()[1].getBuffer();

        int width = imgYUV420.getHeight();
        assert(imgYUV420.getPlanes()[0].getPixelStride() == 1);
        for (int row = imgYUV420.getHeight()-1; row >=0; row--) {
            for (int col = 0; col < imgYUV420.getWidth(); col++) {
                rez[col*width+row] = buffer0.get();
            }
        }
        int uv_offset = imgYUV420.getWidth()*imgYUV420.getHeight();
        assert(imgYUV420.getPlanes()[2].getPixelStride() == imgYUV420.getPlanes()[1].getPixelStride());
        int stride = imgYUV420.getPlanes()[1].getPixelStride();
        for (int row = imgYUV420.getHeight() - 2; row >= 0; row -= 2) {
            for (int col = 0; col < imgYUV420.getWidth(); col += 2) {
                rez[uv_offset+col/2*width+row] = buffer1.get();
                rez[uv_offset+col/2*width+row+1] = buffer2.get();
                for (int skip = 1; skip < stride; skip++) {
                    if (buffer1.remaining() > 0) {
                        buffer1.get();
                    }
                    if (buffer2.remaining() > 0) {
                        buffer2.get();
                    }
                }
            }
        }

        return rez;
    }

    private static byte[] NV21toJPEG(byte[] nv21, int width, int height) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        return out.toByteArray();
    }
}
